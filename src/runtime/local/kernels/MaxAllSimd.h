/*
 * Copyright 2025 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>
#include <chrono>
#include <fstream>
#include <type_traits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg, typename VTRes>
struct MaxAllSimd {
    static VTRes apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function with type checking
// ****************************************************************************

template <class DTArg, typename VTRes>
VTRes maxAllSimd(const DTArg *arg, DCTX(ctx)) {
    // Only use SIMD for float and double types
    if constexpr (std::is_same_v<VTRes, float> || std::is_same_v<VTRes, double>) {
        return MaxAllSimd<DTArg, VTRes>::apply(arg, ctx);
    } else {
        // Fallback to aggAll for all other types
        return aggAll<VTRes, DTArg>(AggOpCode::MAX, arg, ctx);
    }
}

// ****************************************************************************
// (Partial) template specializations for different data types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix<float>
// ----------------------------------------------------------------------------

template <>
struct MaxAllSimd<DenseMatrix<float>, float> {
    __attribute__((target("avx")))
    static float apply(const DenseMatrix<float> *arg, DCTX(ctx)) {        
        // Validation.
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 8)
            throw std::runtime_error(
                "maxAllSimd: for simplicity, the number of cells must be "
                "a multiple of 8"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "maxAllSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );

        // SIMD max reduction (8x f32).
        const float *valuesArg = arg->getValues();
        __m256 vmax = _mm256_loadu_ps(valuesArg);
        valuesArg += 8;
        
        for(size_t i = 8; i < numCells; i += 8) {
            __m256 v = _mm256_loadu_ps(valuesArg);
            vmax = _mm256_max_ps(vmax, v);
            valuesArg += 8;
        }

        // Horizontal max of accumulator elements.
        float M = reinterpret_cast<float*>(&vmax)[0];
        for(int i = 1; i < 8; i++)
            M = std::max(M, reinterpret_cast<float*>(&vmax)[i]);
        
        return M;
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix<double>
// ----------------------------------------------------------------------------

template <>
struct MaxAllSimd<DenseMatrix<double>, double> {
    __attribute__((target("avx")))
    static double apply(const DenseMatrix<double> *arg, DCTX(ctx)) {       
        // Validation.
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells % 4)
            throw std::runtime_error(
                "maxAllSimd: for simplicity, the number of cells must be "
                "a multiple of 4"
            );
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "maxAllSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );

        // SIMD max reduction (4x f64).
        const double *valuesArg = arg->getValues();
        __m256d vmax = _mm256_loadu_pd(valuesArg);
        valuesArg += 4;
        
        for(size_t i = 4; i < numCells; i += 4) {
            __m256d v = _mm256_loadu_pd(valuesArg);
            vmax = _mm256_max_pd(vmax, v);
            valuesArg += 4;
        }
        
        // Horizontal max of accumulator elements.
        double M = reinterpret_cast<double*>(&vmax)[0];
        for(int i = 1; i < 4; i++)
            M = std::max(M, reinterpret_cast<double*>(&vmax)[i]);
        
        return M;
    }
};
