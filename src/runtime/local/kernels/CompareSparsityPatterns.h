#ifndef SRC_RUNTIME_LOCAL_KERNELS_COMPARESPARSITYPATTERNS_H
#define SRC_RUNTIME_LOCAL_KERNELS_COMPARESPARSITYPATTERNS_H

#include <runtime/local/datastructures/CSRMatrix.h>
#include <cstddef>
#include <iostream>

/**
 * @brief Compare the sparsity patterns of two CSR matrices.
 */
template<typename VT>
bool compareSparsityPatterns(const CSRMatrix<VT>* lhs, const CSRMatrix<VT>* rhs) {
    // Check dimensions
    if (lhs->getNumRows() != rhs->getNumRows() || 
        lhs->getNumCols() != rhs->getNumCols()) {
        return false;
    }
    
    // Check nnz
    if (lhs->getNumNonZeros() != rhs->getNumNonZeros()) {
        return false;
    }
    
    const size_t numRows = lhs->getNumRows();
    const size_t nnz = lhs->getNumNonZeros();
    
    // Compare rowOffsets
    const size_t* lhsRowOffsets = lhs->getRowOffsets();
    const size_t* rhsRowOffsets = rhs->getRowOffsets();
    
    for (size_t i = 0; i <= numRows; i++) {
        if (lhsRowOffsets[i] != rhsRowOffsets[i]) {
            return false;
        }
    }
    
    // Compare colIdxs
    const size_t* lhsColIdxs = lhs->getColIdxs();
    const size_t* rhsColIdxs = rhs->getColIdxs();
    
    for (size_t i = 0; i < nnz; i++) {
        if (lhsColIdxs[i] != rhsColIdxs[i]) {
            return false;
        }
    }
    
    return true;
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_COMPARESPARSITYPATTERNS_H