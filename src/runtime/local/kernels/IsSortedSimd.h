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
#include <runtime/local/kernels/IsSorted.h>
#include <ir/daphneir/DataPropertyTypes.h>

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <immintrin.h>
#include <type_traits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg>
struct IsSortedSimd {
    static int64_t apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function with type checking
// ****************************************************************************

template <class DTArg>
int64_t isSortedSimd(const DTArg *arg, DCTX(ctx)) {
    return IsSortedSimd<DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix<double>
// ----------------------------------------------------------------------------

template <>
struct IsSortedSimd<DenseMatrix<double>> {
    __attribute__((target("avx")))
    static int64_t apply(const DenseMatrix<double> *arg, DCTX(ctx)) {
        const size_t rows = arg->getNumRows();
        const size_t cols = arg->getNumCols();
        
        // Validation.
        if(cols != 1)
            throw std::runtime_error(
                "isSortedSimd: for simplicity, only single column matrices are supported"
            );
        if(cols != arg->getRowSkip())
            throw std::runtime_error(
                "isSortedSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );
        if(rows < 4)
            throw std::runtime_error(
                "isSortedSimd: for simplicity, need at least 4 elements for SIMD"
            );
        
        // Handle edge case.
        if(rows == 1)
            return static_cast<int64_t>(MatrixSortness::AllEqual);

        const double *values = arg->getValues();
        
        // Check first few elements to determine direction.
        bool asc = true;
        bool desc = true;
        bool equal = true;
        
        double prev = values[0];
        for(size_t i = 1; i < std::min(rows, size_t(4)); i++) {
            double curr = values[i];
            if(curr != prev) equal = false;
            if(curr < prev) asc = false;
            if(curr > prev) desc = false;
            prev = curr;
        }
        
        // SIMD check for equality (4x f64).
        if(equal) {
            __m256d first = _mm256_broadcast_sd(&values[0]);
            for(size_t i = 0; i + 3 < rows; i += 4) {
                __m256d curr = _mm256_loadu_pd(&values[i]);
                __m256d cmp = _mm256_cmp_pd(curr, first, _CMP_NEQ_OQ);
                if(!_mm256_testz_pd(cmp, cmp)) {
                    equal = false;
                    break;
                }
            }
            // Handle remaining elements.
            for(size_t i = (rows / 4) * 4; i < rows && equal; i++) {
                if(values[i] != values[0]) {
                    equal = false;
                    break;
                }
            }
            if(equal)
                return static_cast<int64_t>(MatrixSortness::AllEqual);
        }
        
        if(!asc && !desc)
            return static_cast<int64_t>(MatrixSortness::NotSorted);
        
        // Check for ascending order.
        if(asc) {
            for(size_t i = 0; i < rows - 1; i++) {
                if(values[i] > values[i + 1]) {
                    asc = false;
                    break;
                }
            }
        }
        
        // Check for descending order.
        if(desc && !asc) {
            for(size_t i = 0; i < rows - 1; i++) {
                if(values[i] < values[i + 1]) {
                    desc = false;
                    break;
                }
            }
        }
        
        if(asc)
            return static_cast<int64_t>(MatrixSortness::SortedAsc);
        else if(desc)
            return static_cast<int64_t>(MatrixSortness::SortedDesc);
        else
            return static_cast<int64_t>(MatrixSortness::NotSorted);
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix<float>
// ----------------------------------------------------------------------------

template <>
struct IsSortedSimd<DenseMatrix<float>> {
    __attribute__((target("avx")))
    static int64_t apply(const DenseMatrix<float> *arg, DCTX(ctx)) {
        const size_t rows = arg->getNumRows();
        const size_t cols = arg->getNumCols();
        
        // Validation.
        if(cols != 1)
            throw std::runtime_error(
                "isSortedSimd: for simplicity, only single column matrices are supported"
            );
        if(cols != arg->getRowSkip())
            throw std::runtime_error(
                "isSortedSimd: for simplicity, the argument must not be "
                "a column segment of another matrix"
            );
        if(rows < 8)
            throw std::runtime_error(
                "isSortedSimd: for simplicity, need at least 8 elements for SIMD"
            );
        
        // Handle edge case.
        if(rows == 1)
            return static_cast<int64_t>(MatrixSortness::AllEqual);

        const float *values = arg->getValues();
        
        // Check first few elements to determine direction.
        bool asc = true;
        bool desc = true;
        bool equal = true;
        
        float prev = values[0];
        for(size_t i = 1; i < std::min(rows, size_t(8)); i++) {
            float curr = values[i];
            if(curr != prev) equal = false;
            if(curr < prev) asc = false;
            if(curr > prev) desc = false;
            prev = curr;
        }
        
        // SIMD check for equality (8x f32).
        if(equal) {
            __m256 first = _mm256_broadcast_ss(&values[0]);
            for(size_t i = 0; i + 7 < rows; i += 8) {
                __m256 curr = _mm256_loadu_ps(&values[i]);
                __m256 cmp = _mm256_cmp_ps(curr, first, _CMP_NEQ_OQ);
                if(!_mm256_testz_ps(cmp, cmp)) {
                    equal = false;
                    break;
                }
            }
            // Handle remaining elements.
            for(size_t i = (rows / 8) * 8; i < rows && equal; i++) {
                if(values[i] != values[0]) {
                    equal = false;
                    break;
                }
            }
            if(equal)
                return static_cast<int64_t>(MatrixSortness::AllEqual);
        }
        
        if(!asc && !desc)
            return static_cast<int64_t>(MatrixSortness::NotSorted);
        
        // Check for ascending order.
        if(asc) {
            for(size_t i = 0; i < rows - 1; i++) {
                if(values[i] > values[i + 1]) {
                    asc = false;
                    break;
                }
            }
        }
        
        // Check for descending order.
        if(desc && !asc) {
            for(size_t i = 0; i < rows - 1; i++) {
                if(values[i] < values[i + 1]) {
                    desc = false;
                    break;
                }
            }
        }
        
        if(asc)
            return static_cast<int64_t>(MatrixSortness::SortedAsc);
        else if(desc)
            return static_cast<int64_t>(MatrixSortness::SortedDesc);
        else
            return static_cast<int64_t>(MatrixSortness::NotSorted);
    }
};