#ifndef PIGGYBACK_ANALYSIS_H
#define PIGGYBACK_ANALYSIS_H

#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>

struct DaphneContext;
template<typename VT> class DenseMatrix;
template<typename VT> class Matrix;

// PiggybackState: tracks analysis state during computation
template<typename VT>
struct PiggybackState {
    // Which properties to analyze
    bool analyzeMinMax;
    bool analyzeSortness;
    bool analyzeSparsity;
    
    // Min/max tracking
    VT minValue;
    VT maxValue;
    
    // Sortness tracking
    bool isSortedAsc;
    bool isSortedDesc;
    bool allEqual;
    VT firstValue;
    VT prevValue;
    bool firstElement;
    
    // Sparsity tracking
    size_t zeroCount;
    size_t totalCount;
    
    // Constructor: initialize everything
    PiggybackState(const std::vector<std::string>& requiredProps) {
        // Default values
        analyzeMinMax = false;
        analyzeSortness = false;
        analyzeSparsity = false;
        minValue = std::numeric_limits<VT>::max();
        maxValue = std::numeric_limits<VT>::lowest();
        isSortedAsc = true;
        isSortedDesc = true;
        allEqual = true;
        firstValue = 0;
        prevValue = 0;
        firstElement = true;
        zeroCount = 0;
        totalCount = 0;
        
        // Check which properties we need
        for(const auto& prop : requiredProps) {
            if(prop == "minmax") analyzeMinMax = true;
            else if(prop == "sortness") analyzeSortness = true;
            else if(prop == "sparsity") analyzeSparsity = true;
        }
    }
    
    // Update analysis with a new value
    inline void update(VT val) {
        // Update min/max
        if(analyzeMinMax) {
            if(firstElement) {
                minValue = val;
                maxValue = val;
            } else {
                if(val < minValue) minValue = val;
                if(val > maxValue) maxValue = val;
            }
        }
        
        // Update sortness
        if(analyzeSortness) {
            if(firstElement) {
                firstValue = val;
                prevValue = val;
            } else {
                if(val != firstValue) allEqual = false;
                if(isSortedAsc || isSortedDesc) {
                    if(val < prevValue) isSortedAsc = false;
                    if(val > prevValue) isSortedDesc = false;
                }
                prevValue = val;
            }
        }
        
        // Update sparsity
        if(analyzeSparsity) {
            if(val == static_cast<VT>(0)) zeroCount++;
            totalCount++;
        }

        firstElement = false;
    }
    
    // Get sortness result
    MatrixSortness getSortness() const {
        if(!analyzeSortness) return MatrixSortness::Unknown;
        
        if(allEqual) return MatrixSortness::AllEqual;
        else if(isSortedAsc) return MatrixSortness::SortedAsc;
        else if(isSortedDesc) return MatrixSortness::SortedDesc;
        else return MatrixSortness::NotSorted;
    }
    
    // Get sparsity result
    double getSparsity() const {
        if(!analyzeSparsity || totalCount == 0) return -1.0;
        size_t nonZeroCount = totalCount - zeroCount;
        return static_cast<double>(nonZeroCount) / static_cast<double>(totalCount);
    }

    // Apply results to matrix
    template<typename MatrixType>
    void applyToMatrix(MatrixType* res) const {
        auto* resMat = const_cast<MatrixType*>(res);
        
        if(analyzeMinMax) {
            resMat->is_minValue = true;
            resMat->minValue = static_cast<double>(minValue);
            resMat->is_maxValue = true;
            resMat->maxValue = static_cast<double>(maxValue);
        }
        
        if(analyzeSortness) {
            resMat->is_sortness = true;
            resMat->sortness = getSortness();
        }
        
        if(analyzeSparsity) {
            resMat->is_sparsity = true;
            resMat->sparsity = getSparsity();
        }
    }
};

/**
 * @brief Get properties to piggyback on the result of a kernel
 *
 * This function determines which properties should be computed (piggybacked) on the
 * result of a kernel operation, based on what downstream operations might need.
 *
 * Modified approach: Always piggyback ALL data properties regardless of kernel type.
 * This ensures maximum compatibility and avoids missing properties that downstream
 * operations might need.
 *
 * @param kernelName The name of the kernel producing the result (e.g., "EwAdd")
 * @param enabled Whether to enable piggyback (default: true)
 * @return Vector of property names that should be piggybacked on the result
 */
inline std::vector<std::string> getPropertiesToPiggyback(
    const std::string& kernelName,
    bool enabled = true
) {
    if(!enabled) {
        return {}; // Piggyback disabled
    }

    // Different operations need different piggyback properties
    else if (kernelName == "EwAdd") {
        // EwAdd operations piggyback all data properties for downstream use
        return {"minmax", "sortness",};
    } else {
        // Default: piggyback basic properties for other operations
        return {"minmax", "sortness", "sparsity"};
    }
}

#endif // PIGGYBACK_ANALYSIS_H

