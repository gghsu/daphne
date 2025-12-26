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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <iostream>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSortnessOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<MatrixSortness> daphne::OrderOp::inferSortness() {
    auto matrixOperand = getOperand(0).getType().dyn_cast<daphne::MatrixType>();

    if (!matrixOperand) {
        return {MatrixSortness::Unknown};
    }

    std::pair<bool, bool> orderOperand = CompilerUtils::isConstant<bool>(getAscs()[0]);

    if (orderOperand.first) {
        bool orderValue = orderOperand.second;
        if (orderValue) {
            return {MatrixSortness::SortedAsc};
        } else {
            return {MatrixSortness::SortedDesc};
        }
    }
    else {
        return {MatrixSortness::Unknown};
    }
}

std::vector<MatrixSortness> daphne::EwLtOp::inferSortness() {
    auto lhsMatrix = getOperand(0).getType().dyn_cast<daphne::MatrixType>();
    std::pair<bool, ssize_t> rhsScalar = CompilerUtils::isConstant<ssize_t>(getOperand(1));
    
    if (!lhsMatrix || !rhsScalar.first) {
        return {MatrixSortness::Unknown};
    }
    
    auto matrixSortness = lhsMatrix.getSortness();
    std::optional<ssize_t> matrixMin = lhsMatrix.getMinValue();
    std::optional<ssize_t> matrixMax = lhsMatrix.getMaxValue();
    ssize_t scalarValue = rhsScalar.second;
    
    if (matrixMin.has_value() && matrixMax.has_value()) {
        if (scalarValue > matrixMax.value() || scalarValue < matrixMin.value()) {
            return {MatrixSortness::AllEqual}; 
        } else {
            if (matrixSortness == MatrixSortness::SortedAsc) {
                return {MatrixSortness::SortedAsc};
            } else if (matrixSortness == MatrixSortness::SortedDesc) {
                return {MatrixSortness::SortedDesc};
            } else {
                //std::cout << "Sortness cannot be retained, returning Unknown" << std::endl;
                return {MatrixSortness::Unknown};
            }  
        }
    } else {
        return {MatrixSortness::Unknown};
    }
}

std::vector<MatrixSortness> daphne::EwAddOp::inferSortness() {

    auto lhsMatrix = getOperand(0).getType().dyn_cast<daphne::MatrixType>();
    auto rhsMatrix = getOperand(1).getType().dyn_cast<daphne::MatrixType>();

    if (!lhsMatrix || !rhsMatrix) {
        return {MatrixSortness::Unknown};
    }

    auto lhsSortness = lhsMatrix.getSortness();
    auto rhsSortness = rhsMatrix.getSortness();

    // Check if both matrices are sorted
    if ((lhsSortness == MatrixSortness::SortedAsc || lhsSortness == MatrixSortness::SortedDesc) &&
        (rhsSortness == MatrixSortness::SortedAsc || rhsSortness == MatrixSortness::SortedDesc)) {
            // If both matrices are sorted in ascending order, result is SortedAsc
            if (lhsSortness == MatrixSortness::SortedAsc && rhsSortness == MatrixSortness::SortedAsc) {
                return {MatrixSortness::SortedAsc};
            }
            // If both matrices are sorted in descending order, result is SortedDesc
            if (lhsSortness == MatrixSortness::SortedDesc && rhsSortness == MatrixSortness::SortedDesc) {
                return {MatrixSortness::SortedDesc};
            }
    }

    return {MatrixSortness::Unknown};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<MatrixSortness> daphne::tryInferSortness(Operation *op) {
    if (auto inferSortnessOp = llvm::dyn_cast<daphne::InferSortness>(op))
        // If the operation implements the inference interface, we apply that.
        return inferSortnessOp.inferSortness();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<MatrixSortness> sortnesses;
        for (size_t i = 0; i < op->getNumResults(); i++)
            sortnesses.push_back(MatrixSortness::Unknown);
        return sortnesses;
    }
}