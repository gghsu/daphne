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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expargs or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
#define SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H

#include <cstddef>
#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <ir/daphneir/DataPropertyTypes.h>
#include <unordered_set>
#include <limits>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct RecordProperties {
    static void apply(const DT *arg, uint32_t valueId, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT> void recordProperties(const DT *arg, uint32_t valueId, DCTX(ctx)) {
    RecordProperties<DT>::apply(arg, valueId, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix Record Implementation
// ----------------------------------------------------------------------------

template <typename VT> struct RecordProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, uint32_t valueId, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        size_t nnz = 0;
        std::unordered_set<VT> distinctValues;
        double minValue = std::numeric_limits<double>::max();
        double maxValue = std::numeric_limits<double>::lowest();
        MatrixSortness sortness = MatrixSortness::Unknown;
        
        // ollect statistics
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                VT val = arg->get(r, c);
                if (val != 0) {
                    nnz++;
                }
                distinctValues.insert(val);
                minValue = std::min(minValue, static_cast<double>(val));
                maxValue = std::max(maxValue, static_cast<double>(val));
            }
        }

        // Check sortness (column-wise)
        if (numRows > 1 && numCols > 0) {
            bool isAscending = true;
            bool isDescending = true;
            bool isAllEqual = true;
            
            VT prev = arg->get(0, 0);
            
            for (size_t r = 1; r < numRows; r++) {
                VT curr = arg->get(r, 0); 
                
                if (curr != prev)
                    isAllEqual = false;
                if (curr < prev)
                    isAscending = false;
                if (curr > prev)
                    isDescending = false;
                    
                // Early termination
                if (!isAscending && !isDescending && !isAllEqual)
                    break;
                    
                prev = curr;
            }
            
            if (isAllEqual) {
                sortness = MatrixSortness::AllEqual;
            } else if (isAscending) {
                sortness = MatrixSortness::SortedAsc;
            } else if (isDescending) {
                sortness = MatrixSortness::SortedDesc;
            } else {
                sortness = MatrixSortness::NotSorted;
            }
        } else if (numRows <= 1) {
            sortness = MatrixSortness::AllEqual;
        }

        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);
        const int64_t distinct = static_cast<int64_t>(distinctValues.size());
        
        ctx->propertyLogger.logProperty(valueId, std::make_unique<SparsityProperty>(sparsity));
        ctx->propertyLogger.logProperty(valueId, std::make_unique<DistinctProperty>(distinct));
        ctx->propertyLogger.logProperty(valueId, std::make_unique<MinValueProperty>(minValue));
        ctx->propertyLogger.logProperty(valueId, std::make_unique<MaxValueProperty>(maxValue));
        ctx->propertyLogger.logProperty(valueId, std::make_unique<SortnessProperty>(static_cast<int64_t>(sortness)));
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix Record Implementation
// ----------------------------------------------------------------------------

template <typename VT> struct RecordProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, uint32_t valueId, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        size_t nnz = 0;
        const VT *values = arg->getValues();
        // Even a matrix in CSR representation might store some zero values explicitly. Thus, we check for non-zeros
        // here.
        for (size_t i = 0; i < arg->getNumNonZeros(); i++)
            if (values[i] != 0)
                nnz++;

        const double sparsity = static_cast<double>(nnz) / (numRows * numCols);
        ctx->propertyLogger.logProperty(valueId, std::make_unique<SparsityProperty>(sparsity));
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_RECORD_PROPERTIES_H
