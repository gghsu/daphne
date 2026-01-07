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
struct MinAllSimd {
    static VTRes apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function with type checking
// ****************************************************************************

template <class DTArg, typename VTRes>
VTRes minAllSimd(const DTArg *arg, DCTX(ctx)) {
    // Only use SIMD for float and double types
    if constexpr (std::is_same_v<VTRes, float> || std::is_same_v<VTRes, double>) {
        return MinAllSimd<DTArg, VTRes>::apply(arg, ctx);
    } else {
        // Fallback to aggAll for all other types
        return aggAll<VTRes, DTArg>(AggOpCode::MIN, arg, ctx);
    }
}

// ****************************************************************************
// (Partial) template specializations for different data types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix<float>
// ----------------------------------------------------------------------------

template <>
struct MinAllSimd<DenseMatrix<float>, float> {
    __attribute__((target("avx")))
    static float apply(const DenseMatrix<float> *arg, DCTX(ctx)) {       
        // Validation.
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells == 0)
            throw std::runtime_error("minAllSimd: matrix must not be empty");
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "minAllSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );

        const float *valuesArg = arg->getValues();
        const size_t simdWidth = 8;
        const size_t numSIMDIters = numCells / simdWidth;
        const size_t tailStart = numSIMDIters * simdWidth;
        
        // SIMD min reduction (8x f32).
        __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::max());
        
        for(size_t i = 0; i < numSIMDIters; i++) {
            __m256 v = _mm256_loadu_ps(valuesArg + i * simdWidth);
            vmin = _mm256_min_ps(vmin, v);
        }
        
        // Horizontal min of accumulator elements.
        float m = reinterpret_cast<float*>(&vmin)[0];
        for(int i = 1; i < 8; i++)
            m = std::min(m, reinterpret_cast<float*>(&vmin)[i]);
        
        // Process tail elements (if any).
        for(size_t i = tailStart; i < numCells; i++)
            m = std::min(m, valuesArg[i]);
        
        return m;
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix<double>
// ----------------------------------------------------------------------------

template <>
struct MinAllSimd<DenseMatrix<double>, double> {
    __attribute__((target("avx")))
    static double apply(const DenseMatrix<double> *arg, DCTX(ctx)) {
       // Validation.
        const size_t numCells = arg->getNumRows() * arg->getNumCols();
        if(numCells == 0)
            throw std::runtime_error("minAllSimd: matrix must not be empty");
        if(arg->getNumCols() != arg->getRowSkip())
            throw std::runtime_error(
                "minAllSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );

        const double *valuesArg = arg->getValues();
        const size_t simdWidth = 4;
        const size_t numSIMDIters = numCells / simdWidth;
        const size_t tailStart = numSIMDIters * simdWidth;
        
        // SIMD min reduction (4x f64).
        __m256d vmin = _mm256_set1_pd(std::numeric_limits<double>::max());
        
        for(size_t i = 0; i < numSIMDIters; i++) {
            __m256d v = _mm256_loadu_pd(valuesArg + i * simdWidth);
            vmin = _mm256_min_pd(vmin, v);
        }
        
        // Horizontal min of accumulator elements.
        double m = reinterpret_cast<double*>(&vmin)[0];
        for(int i = 1; i < 4; i++)
            m = std::min(m, reinterpret_cast<double*>(&vmin)[i]);
        
        // Process tail elements (if any).
        for(size_t i = tailStart; i < numCells; i++)
            m = std::min(m, valuesArg[i]);
        
        return m;
    }
};
