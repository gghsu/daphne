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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cstddef>
#include <cstdio>

template<class DTArg>
struct IsSorted {
    static int64_t apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTArg>
int64_t isSorted(const DTArg *arg, DCTX(ctx)) {
    return IsSorted<DTArg>::apply(arg, ctx);
}

// ****************************************************************************
// Template specialization for DenseMatrix
// ****************************************************************************

template<typename VT>
struct IsSorted<DenseMatrix<VT>> {
    static int64_t apply(const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        // Handle empty or single-element cases
        if(numRows <= 1)
            return static_cast<int64_t>(MatrixSortness::AllEqual);
        
        bool isAscending = true;
        bool isDescending = true;
        bool isAllEqual = true;
        
        const VT *values = arg->getValues();
        const size_t rowSkip = arg->getRowSkip();
        VT prev = values[0];  // First element (row 0, col 0)
        
        for(size_t i = 1; i < numRows; i++) {
            VT curr = values[i * rowSkip];  // First column of row i
            
            if(curr != prev)
                isAllEqual = false;
            if(curr < prev)
                isAscending = false;
            if(curr > prev)
                isDescending = false;
                
            // Early termination
            if(!isAscending && !isDescending && !isAllEqual)
                return static_cast<int64_t>(MatrixSortness::NotSorted);
                
            prev = curr;
        }
        
        if(isAllEqual)
            return static_cast<int64_t>(MatrixSortness::AllEqual);
        if(isAscending)
            return static_cast<int64_t>(MatrixSortness::SortedAsc);
        if(isDescending)
            return static_cast<int64_t>(MatrixSortness::SortedDesc);
            
        return static_cast<int64_t>(MatrixSortness::NotSorted);
    }
};