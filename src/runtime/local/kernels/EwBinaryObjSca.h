/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cstddef>
#include <cstring>
#include <iomanip>  // for std::fixed, std::setprecision
#include <algorithm>
#include <type_traits>
#include <chrono> // for runtime measurement
#include <fstream>
#include <atomic>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, typename VTRhs> struct EwBinaryObjSca {
    static void apply(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, VTRhs rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, typename VTRhs>
void ewBinaryObjSca(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, VTRhs rhs, DCTX(ctx)) {
    EwBinaryObjSca<DTRes, DTLhs, VTRhs>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, scalar
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTLhs, typename VTRhs>
struct EwBinaryObjSca<DenseMatrix<VTRes>, DenseMatrix<VTLhs>, VTRhs> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTRes> *&res, const DenseMatrix<VTLhs> *lhs, VTRhs rhs,
                      DCTX(ctx)) {
        // Start timing for performance measurement
        auto startTime = std::chrono::high_resolution_clock::now();
        
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        // Get matrix properties for optimization
        MatrixSortness sortness = lhs->sortness;
        bool hasMin = lhs->is_minValue;
        bool hasMax = lhs->is_maxValue;
        double minValue = lhs->minValue;
        double maxValue = lhs->maxValue;
        
        bool isSorted = (sortness == MatrixSortness::SortedAsc || sortness == MatrixSortness::SortedDesc);

        // Try optimization 1: Bounds-based shortcut for comparison operations
        if constexpr (std::is_arithmetic_v<VTLhs> && std::is_arithmetic_v<VTRhs>) {
            if (applyBoundsOptimization(opCode, res, hasMin, hasMax, minValue, maxValue, rhs, numRows, numCols)) {
                auto endTime = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double>(endTime - startTime).count();
                std::cerr << "[KERNEL_TIME] EwBinaryObjSca: " << std::fixed << std::setprecision(6)
                          << seconds << " seconds (bounds_optimization)" << std::endl;
                return;
            }
        }

        // Try optimization 2: Binary search for sorted data
        if (isSorted && isComparisonOp(opCode)) {
            applySortedOptimization(opCode, lhs, res, rhs, sortness, numRows, numCols, ctx);
            auto endTime = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(endTime - startTime).count();
            std::cerr << "[KERNEL_TIME] EwBinaryObjSca: " << std::fixed << std::setprecision(6)
                      << seconds << " seconds (binary_search)" << std::endl;
        } else {
            // Fallback: Process each element one by one
            applyElementWise(opCode, lhs, res, rhs, numRows, numCols, ctx);
            auto endTime = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(endTime - startTime).count();
            std::cerr << "[KERNEL_TIME] EwBinaryObjSca: " << std::fixed << std::setprecision(6)
                      << seconds << " seconds (elementwise)" << std::endl;
        }
    }

private:
    static bool isComparisonOp(BinaryOpCode opCode) {
        switch (opCode) {
            case BinaryOpCode::LT:
            case BinaryOpCode::LE:
            case BinaryOpCode::GT:
            case BinaryOpCode::GE:
            case BinaryOpCode::EQ:
            case BinaryOpCode::NEQ:
                return true;
            default:
                return false;
        }
    }

    static bool applyBoundsOptimization(BinaryOpCode opCode, DenseMatrix<VTRes> *res, 
                                    bool hasMin, bool hasMax, double minValue, double maxValue,
                                    VTRhs rhs, size_t numRows, size_t numCols) {
        double rhsValue = static_cast<double>(rhs);
        VTRes *valuesRes = res->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        
        // Fast fill: result is always 0 or 1 for comparison operations
        auto fillMatrix = [&](bool value) {
            const VTRes fillVal = static_cast<VTRes>(value ? 1 : 0);
            if (rowSkipRes == numCols) {
                // Contiguous memory: can use optimized filling
                VTRes *end = valuesRes + (numRows * numCols);
                std::fill(valuesRes, end, fillVal);
            } else {
                // Non-contiguous: fill row by row
                VTRes *p = valuesRes;
                for (size_t r = 0; r < numRows; ++r) {
                    std::fill(p, p + numCols, fillVal);
                    p += rowSkipRes;
                }
            }
            return true;
        };
        
        // Case 1: All elements have the same value
        if (hasMin && hasMax && minValue == maxValue) {
            switch (opCode) {
                case BinaryOpCode::LT:  return fillMatrix(minValue < rhsValue);
                case BinaryOpCode::LE:  return fillMatrix(minValue <= rhsValue);
                case BinaryOpCode::GT:  return fillMatrix(minValue > rhsValue);
                case BinaryOpCode::GE:  return fillMatrix(minValue >= rhsValue);
                case BinaryOpCode::EQ:  return fillMatrix(minValue == rhsValue);
                case BinaryOpCode::NEQ: return fillMatrix(minValue != rhsValue);
                default: return false;
            }
        }
        
        // Case 2: Range-based shortcuts
        switch (opCode) {
            case BinaryOpCode::LT:
                if (hasMax && maxValue < rhsValue) return fillMatrix(true);
                if (hasMin && minValue >= rhsValue) return fillMatrix(false);
                break;
            case BinaryOpCode::LE:
                if (hasMax && maxValue <= rhsValue) return fillMatrix(true);
                if (hasMin && minValue > rhsValue) return fillMatrix(false);
                break;
            case BinaryOpCode::GT:
                if (hasMin && minValue > rhsValue) return fillMatrix(true);
                if (hasMax && maxValue <= rhsValue) return fillMatrix(false);
                break;
            case BinaryOpCode::GE:
                if (hasMin && minValue >= rhsValue) return fillMatrix(true);
                if (hasMax && maxValue < rhsValue) return fillMatrix(false);
                break;
            case BinaryOpCode::EQ:
                if (hasMin && hasMax && (rhsValue < minValue || rhsValue > maxValue)) {
                    return fillMatrix(false);
                }
                break;
            case BinaryOpCode::NEQ:
                if (hasMin && hasMax && (rhsValue < minValue || rhsValue > maxValue)) {
                    return fillMatrix(true);
                }
                break;
            default:
                break;
        }
        return false;
    }

    static void applySortedOptimization(BinaryOpCode opCode, const DenseMatrix<VTLhs> *lhs,
                                      DenseMatrix<VTRes> *res, VTRhs rhs, MatrixSortness sortness,
                                      size_t numRows, size_t numCols, DCTX(ctx)) {
        const VTLhs *valuesLhs = lhs->getValues();
        VTRes *valuesRes = res->getValues();
        const size_t rowSkipLhs = lhs->getRowSkip();
        const size_t rowSkipRes = res->getRowSkip();

        const bool isAscending = (sortness == MatrixSortness::SortedAsc);
        const bool isDescending = (sortness == MatrixSortness::SortedDesc);

        if (!isAscending && !isDescending) {
            // Not sorted, fall back to element-wise
            applyElementWise(opCode, lhs, res, rhs, numRows, numCols, ctx);
            return;
        }

        for (size_t j = 0; j < numCols; ++j) {
            const VTLhs *colData = valuesLhs + j;
            VTRes *resColData = valuesRes + j;

            size_t splitPos = 0;
            bool beforeSplit = false;
            bool afterSplit = false;

            if (isAscending) {
                switch (opCode) {
                    case BinaryOpCode::LT: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] < rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = true;
                        afterSplit = false;
                        break;
                    }
                    case BinaryOpCode::LE: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] <= rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = true;
                        afterSplit = false;
                        break;
                    }
                    case BinaryOpCode::GT: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] <= rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = false;
                        afterSplit = true;
                        break;
                    }
                    case BinaryOpCode::GE: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] < rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = false;
                        afterSplit = true;
                        break;
                    }
                    default:
                        applyElementWise(opCode, lhs, res, rhs, numRows, numCols, ctx);
                        return;
                }
            } else {
                switch (opCode) {
                    case BinaryOpCode::LT: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] > rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = false;
                        afterSplit = true;
                        break;
                    }
                    case BinaryOpCode::LE: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] >= rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = false;
                        afterSplit = true;
                        break;
                    }
                    case BinaryOpCode::GT: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] >= rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = true;
                        afterSplit = false;
                        break;
                    }
                    case BinaryOpCode::GE: {
                        size_t left = 0, right = numRows;
                        while (left < right) {
                            size_t mid = left + (right - left) / 2;
                            if (colData[mid * rowSkipLhs] > rhs) left = mid + 1;
                            else right = mid;
                        }
                        splitPos = left;
                        beforeSplit = true;
                        afterSplit = false;
                        break;
                    }
                    default:
                        applyElementWise(opCode, lhs, res, rhs, numRows, numCols, ctx);
                        return;
                }
            }

            if constexpr (std::is_same_v<VTRes, bool> || std::is_arithmetic_v<VTRes>) {
                const VTRes beforeVal = static_cast<VTRes>(beforeSplit ? 1 : 0);
                const VTRes afterVal = static_cast<VTRes>(afterSplit ? 1 : 0);

                for (size_t r = 0; r < splitPos; r++) {
                    resColData[r * rowSkipRes] = beforeVal;
                }
                for (size_t r = splitPos; r < numRows; r++) {
                    resColData[r * rowSkipRes] = afterVal;
                }
            }
        }
    }

    static void applyElementWise(BinaryOpCode opCode, const DenseMatrix<VTLhs> *lhs, 
                               DenseMatrix<VTRes> *res, VTRhs rhs, 
                               size_t numRows, size_t numCols, DCTX(ctx)) {
        const VTLhs *valuesLhs = lhs->getValues();
        VTRes *valuesRes = res->getValues();
        EwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs> func = getEwBinaryScaFuncPtr<VTRes, VTLhs, VTRhs>(opCode);

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesLhs[c], rhs, ctx);
            valuesLhs += lhs->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, scalar
// ----------------------------------------------------------------------------

template <typename VT> struct EwBinaryObjSca<Matrix<VT>, Matrix<VT>, VT> {
    static void apply(BinaryOpCode opCode, Matrix<VT> *&res, const Matrix<VT> *lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        // TODO Choose matrix implementation depending on expected number of
        // non-zeros.
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(lhs->get(r, c), rhs, ctx));
        res->finishAppend();
    }
};

// ----------------------------------------------------------------------------
// Frame <- Frame, scalar
// ----------------------------------------------------------------------------

template <typename VT>
void ewBinaryFrameColSca(BinaryOpCode opCode, Frame *&res, const Frame *lhs, VT rhs, size_t c, DCTX(ctx)) {
    auto *col_res = res->getColumn<VT>(c);
    auto *col_lhs = lhs->getColumn<VT>(c);
    ewBinaryObjSca<DenseMatrix<VT>, DenseMatrix<VT>, VT>(opCode, col_res, col_lhs, rhs, ctx);
}

template <typename VT> struct EwBinaryObjSca<Frame, Frame, VT> {
    static void apply(BinaryOpCode opCode, Frame *&res, const Frame *lhs, VT rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<Frame>(numRows, numCols, lhs->getSchema(), lhs->getLabels(), false);

        for (size_t c = 0; c < numCols; c++) {
            switch (lhs->getColumnType(c)) {
            // For all value types:
            case ValueTypeCode::F64:
                ewBinaryFrameColSca<double>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::F32:
                ewBinaryFrameColSca<float>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI64:
                ewBinaryFrameColSca<int64_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI32:
                ewBinaryFrameColSca<int32_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::SI8:
                ewBinaryFrameColSca<int8_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI64:
                ewBinaryFrameColSca<uint64_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI32:
                ewBinaryFrameColSca<uint32_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            case ValueTypeCode::UI8:
                ewBinaryFrameColSca<uint8_t>(opCode, res, lhs, rhs, c, ctx);
                break;
            default:
                throw std::runtime_error("EwBinaryObjSca::apply: unknown value type code");
            }
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_EWBINARYOBJSCA_H
