#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <immintrin.h> // for the SIMD-enabled kernel
#include <iostream>
#include <iomanip>
#include <cblas.h>
#include <stdexcept>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <cmath>

#include "PiggybackAnalysis.h"

// Forward declaration - we don't include DaphneContext.h to avoid LLVM dependencies
class DaphneContext;

extern "C" {
    // Custom sequential sum-kernel.
    void mySumSeq(
        float * res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        std::cerr << "hello from mySumSeq()" << std::endl;
        
        // Get properties to piggyback on the RESULT for downstream operations
        auto propsToCompute = getPropertiesToPiggyback("Sum", true);
        PiggybackState<float> piggyback(propsToCompute);

        const float * valuesArg = arg->getValues();
        *res = 0;
        for(size_t r = 0; r < arg->getNumRows(); r++) {
            for(size_t c = 0; c < arg->getNumCols(); c++) {
                float val = valuesArg[c];
                *res += val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
        }
    }
    
    // Custom SIMD-enabled sum-kernel.
    void mySumSIMD(
        float * res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        std::cerr << "hello from mySumSIMD()" << std::endl;

        // Validation.
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 8)
            throw std::runtime_error(
                "for simplicity, the number of cells must be "
                "a multiple of 8"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must not be "
                "a column segment of another matrix"
            );
        
        // SIMD accumulation (8x f32).
        const float * valuesArg = arg->getValues();
        __m256 acc = _mm256_setzero_ps();
        for(size_t i = 0; i < numCells / 8; i++) {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(valuesArg));
            valuesArg += 8;
        }
        
        // Summation of accumulator elements.
        *res =
            (reinterpret_cast<float*>(&acc))[0] +
            (reinterpret_cast<float*>(&acc))[1] +
            (reinterpret_cast<float*>(&acc))[2] +
            (reinterpret_cast<float*>(&acc))[3] +
            (reinterpret_cast<float*>(&acc))[4] +
            (reinterpret_cast<float*>(&acc))[5] +
            (reinterpret_cast<float*>(&acc))[6] +
            (reinterpret_cast<float*>(&acc))[7];
    }

    // Custom SIMD-enabled min-kernel (f32).
    void myMinSIMD(
        float * res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 8)
            throw std::runtime_error(
                "for simplicity, the number of cells must be a multiple of 8"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must be contiguous (no col segment)"
            );

        const float * values = arg->getValues();
        __m256 vmin = _mm256_loadu_ps(values);
        values += 8;
        for(size_t i = 8; i < numCells; i += 8) {
            __m256 v = _mm256_loadu_ps(values);
            vmin = _mm256_min_ps(vmin, v);
            values += 8;
        }
        // Horizontal min of vmin lanes
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, vmin);
        float m = lanes[0];
        for(int i = 1; i < 8; i++) m = std::min(m, lanes[i]);
        *res = m;
    }

    // Custom SIMD-enabled max-kernel (f32).
    void myMaxSIMD(
        float * res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 8)
            throw std::runtime_error(
                "for simplicity, the number of cells must be a multiple of 8"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must be contiguous (no col segment)"
            );

        const float * values = arg->getValues();
        __m256 vmax = _mm256_loadu_ps(values);
        values += 8;
        for(size_t i = 8; i < numCells; i += 8) {
            __m256 v = _mm256_loadu_ps(values);
            vmax = _mm256_max_ps(vmax, v);
            values += 8;
        }
        // Horizontal max of vmax lanes
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, vmax);
        float M = lanes[0];
        for(int i = 1; i < 8; i++) M = std::max(M, lanes[i]);
        *res = M;
    }

    // Custom SIMD-enabled min-kernel (f64).
    void myMinSIMD64(
        double * res,
        const DenseMatrix<double> * arg,
        DaphneContext * ctx
    ) {
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 4)
            throw std::runtime_error(
                "for simplicity, the number of cells must be a multiple of 4"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must be contiguous (no col segment)"
            );

        const double * values = arg->getValues();
        __m256d vmin = _mm256_loadu_pd(values);
        values += 4;
        for(size_t i = 4; i < numCells; i += 4) {
            __m256d v = _mm256_loadu_pd(values);
            vmin = _mm256_min_pd(vmin, v);
            values += 4;
        }
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, vmin);
        double m = lanes[0];
        for(int i = 1; i < 4; i++) m = std::min(m, lanes[i]);
        *res = m;
    }

    // Custom SIMD-enabled max-kernel (f64).
    void myMaxSIMD64(
        double * res,
        const DenseMatrix<double> * arg,
        DaphneContext * ctx
    ) {
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 4)
            throw std::runtime_error(
                "for simplicity, the number of cells must be a multiple of 4"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "for simplicity, the argument must be contiguous (no col segment)"
            );

        const double * values = arg->getValues();
        __m256d vmax = _mm256_loadu_pd(values);
        values += 4;
        for(size_t i = 4; i < numCells; i += 4) {
            __m256d v = _mm256_loadu_pd(values);
            vmax = _mm256_max_pd(vmax, v);
            values += 4;
        }
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, vmax);
        double M = lanes[0];
        for(int i = 1; i < 4; i++) M = std::max(M, lanes[i]);
        *res = M;
    }

    // Custom element-wise addition kernel for float with automated piggyback analysis
    void myEwAdd(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * lhs,
        const DenseMatrix<float> * rhs,
        DaphneContext * ctx
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        // Check dimension compatibility
        if(numRowsLhs != numRowsRhs || numColsLhs != numColsRhs) {
            throw std::runtime_error("myEwAdd: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRowsLhs, numColsLhs, false);
        }

        const float * valuesLhs = lhs->getValues();
        const float * valuesRhs = rhs->getValues();
        float * valuesRes = res->getValues();

        // Get properties to piggyback on the RESULT for downstream operations
        // Default: always enable piggyback analysis
        auto propsToCompute = getPropertiesToPiggyback("EwAdd", true);
        PiggybackState<float> piggyback(propsToCompute);

        // Element-wise addition with automated piggyback analysis
        for(size_t r = 0; r < numRowsLhs; r++) {
            for(size_t c = 0; c < numColsLhs; c++) {
                float val = valuesLhs[c] + valuesRhs[c];
                valuesRes[c] = val;

                // Automated piggyback: only analyzes properties specified in adaptive_map
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cerr << "*** myEwAdd (float) execution time: " << duration.count() << " seconds ***" << std::endl;
    }

    // Custom element-wise addition kernel for double with automated piggyback analysis
    void myEwAddDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * lhs,
        const DenseMatrix<double> * rhs,
        DaphneContext * ctx
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        // Check dimension compatibility
        if(numRowsLhs != numRowsRhs || numColsLhs != numColsRhs) {
            throw std::runtime_error("myEwAddDouble: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRowsLhs, numColsLhs, false);
        }

        const double * valuesLhs = lhs->getValues();
        const double * valuesRhs = rhs->getValues();
        double * valuesRes = res->getValues();

        // Get properties to piggyback on the RESULT for downstream operations
        // Default: always enable piggyback analysis
        auto propsToCompute = getPropertiesToPiggyback("EwAdd", true);
        PiggybackState<double> piggyback(propsToCompute);
        for(size_t r = 0; r < numRowsLhs; r++) {
            for(size_t c = 0; c < numColsLhs; c++) {
                double val = valuesLhs[c] + valuesRhs[c];
                valuesRes[c] = val;

                // Automated piggyback: only analyzes properties specified in adaptive_map
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cerr << "*** myEwAddDouble execution time: " << duration.count() << " seconds ***" << std::endl;
    }

    // Custom element-wise addition kernel for int64_t with automated piggyback analysis
    void myEwAddInt64(
        DenseMatrix<int64_t> * &res,
        const DenseMatrix<int64_t> * lhs,
        const DenseMatrix<int64_t> * rhs,
        DaphneContext * ctx
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        if(numRowsLhs != numRowsRhs || numColsLhs != numColsRhs) {
            throw std::runtime_error("myEwAddInt64: Matrix dimensions must match for element-wise addition");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<int64_t>>(numRowsLhs, numColsLhs, false);
        }

        const int64_t * valuesLhs = lhs->getValues();
        const int64_t * valuesRhs = rhs->getValues();
        int64_t * valuesRes = res->getValues();

        // Get properties to piggyback on the RESULT for downstream operations
        // Default: always enable piggyback analysis
        auto propsToCompute = getPropertiesToPiggyback("EwAdd", true);
        PiggybackState<int64_t> piggyback(propsToCompute);
        for(size_t r = 0; r < numRowsLhs; r++) {
            for(size_t c = 0; c < numColsLhs; c++) {
                int64_t val = valuesLhs[c] + valuesRhs[c];
                valuesRes[c] = val;

                // Automated piggyback: only analyzes properties specified in adaptive_map
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cerr << "*** myEwAddInt64 execution time: " << duration.count() << " seconds ***" << std::endl;
    }

    // ========== Element-wise Subtraction (EwSub) with Piggyback ==========

    void myEwSub(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * lhs,
        const DenseMatrix<float> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwSub: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesLhs = lhs->getValues();
        const float * valuesRhs = rhs->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwSub", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = valuesLhs[c] - valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwSubDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * lhs,
        const DenseMatrix<double> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwSubDouble: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesLhs = lhs->getValues();
        const double * valuesRhs = rhs->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwSub", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = valuesLhs[c] - valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    // ========== Element-wise Multiplication (EwMul) with Piggyback ==========

    void myEwMul(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * lhs,
        const DenseMatrix<float> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwMul: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesLhs = lhs->getValues();
        const float * valuesRhs = rhs->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwMul", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = valuesLhs[c] * valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwMulDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * lhs,
        const DenseMatrix<double> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwMulDouble: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesLhs = lhs->getValues();
        const double * valuesRhs = rhs->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwMul", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = valuesLhs[c] * valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    // ========== Element-wise Division (EwDiv) with Piggyback ==========

    void myEwDiv(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * lhs,
        const DenseMatrix<float> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwDiv: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesLhs = lhs->getValues();
        const float * valuesRhs = rhs->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwDiv", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = valuesLhs[c] / valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwDivDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * lhs,
        const DenseMatrix<double> * rhs,
        DaphneContext * ctx
    ) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if(numRows != rhs->getNumRows() || numCols != rhs->getNumCols()) {
            throw std::runtime_error("myEwDivDouble: lhs and rhs must have the same dimensions");
        }

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesLhs = lhs->getValues();
        const double * valuesRhs = rhs->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwDiv", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = valuesLhs[c] / valuesRhs[c];
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesLhs += lhs->getRowSkip();
            valuesRhs += rhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    // ========== Element-wise Square Root (EwSqrt) with Piggyback ==========

    void myEwSqrt(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesArg = arg->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwSqrt", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = std::sqrt(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwSqrtDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesArg = arg->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwSqrt", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = std::sqrt(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    // ========== Element-wise Absolute Value (EwAbs) with Piggyback ==========

    void myEwAbs(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesArg = arg->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwAbs", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = std::abs(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwAbsDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesArg = arg->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwAbs", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = std::abs(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    // ========== Element-wise Exponential (EwExp) with Piggyback ==========

    void myEwExp(
        DenseMatrix<float> * &res,
        const DenseMatrix<float> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<float>>(numRows, numCols, false);
        }

        const float * valuesArg = arg->getValues();
        float * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwExp", true);
        PiggybackState<float> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                float val = std::exp(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }

    void myEwExpDouble(
        DenseMatrix<double> * &res,
        const DenseMatrix<double> * arg,
        DaphneContext * ctx
    ) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if(res == nullptr) {
            res = DataObjectFactory::create<DenseMatrix<double>>(numRows, numCols, false);
        }

        const double * valuesArg = arg->getValues();
        double * valuesRes = res->getValues();

        auto propsToCompute = getPropertiesToPiggyback("EwExp", true);
        PiggybackState<double> piggyback(propsToCompute);

        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++) {
                double val = std::exp(valuesArg[c]);
                valuesRes[c] = val;
                piggyback.update(val);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }

        piggyback.applyToMatrix(res);
    }
}