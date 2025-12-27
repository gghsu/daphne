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

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferMinMaxOpInterface.cpp.inc>
}

#include <vector>
#include <optional>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************
std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::RandMatrixOp::inferMinMax() {
    // Try to get min/max as double constants
    auto minC = CompilerUtils::isConstant<double>(getMin());
    auto maxC = CompilerUtils::isConstant<double>(getMax());
    if (minC.first && maxC.first)
        return {{minC.second, maxC.second}};
    return {{std::nullopt, std::nullopt}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::FillOp::inferMinMax() {
    // The fill value can be any numeric scalar; infer min=max=value if constant
    auto fillC = CompilerUtils::isConstant<double>(getArg());
    if (fillC.first)
        return {{fillC.second, fillC.second}};
    return {{std::nullopt, std::nullopt}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::OrderOp::inferMinMax() {
    // Ordering does not change the set of values; propagate input min/max if known
    auto mt = getArg().getType().dyn_cast<daphne::MatrixType>();
    if (!mt)
        return {{std::nullopt, std::nullopt}};
    return {{mt.getMinValue(), mt.getMaxValue()}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::EwLtOp::inferMinMax() {
    // Element-wise less-than with scalar RHS produces boolean 0/1
    auto lhsMatrix = getOperand(0).getType().dyn_cast<daphne::MatrixType>();
    auto rhsC = CompilerUtils::isConstant<double>(getOperand(1));
    if (!lhsMatrix || !rhsC.first)
        return {{std::nullopt, std::nullopt}};

    auto matrixMin = lhsMatrix.getMinValue();
    auto matrixMax = lhsMatrix.getMaxValue();
    double scalarValue = rhsC.second;

    if (matrixMin.has_value() && matrixMax.has_value()) {
        if (scalarValue > matrixMax.value())
            return {{1.0, 1.0}}; // all true
        else if (scalarValue <= matrixMin.value())
            return {{0.0, 0.0}}; // all false (<= because x < min => always false)
        else
            return {{0.0, 1.0}}; // mixed
    }
    return {{std::nullopt, std::nullopt}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::EwNeqOp::inferMinMax() {
    // Element-wise not-equal with scalar RHS produces boolean 0/1
    auto lhsMatrix = getOperand(0).getType().dyn_cast<daphne::MatrixType>();
    auto rhsC = CompilerUtils::isConstant<double>(getOperand(1));
    if (!lhsMatrix || !rhsC.first)
        return {{std::nullopt, std::nullopt}};

    auto matrixMin = lhsMatrix.getMinValue();
    auto matrixMax = lhsMatrix.getMaxValue();
    double scalarValue = rhsC.second;

    if (matrixMin.has_value() && matrixMax.has_value()) {
        if (scalarValue > matrixMax.value() || scalarValue < matrixMin.value())
            return {{1.0, 1.0}}; // always not equal
        // If both min and max equal the scalar, then all entries equal scalar
        if (matrixMin.value() == scalarValue && matrixMax.value() == scalarValue)
            return {{0.0, 0.0}}; // always equal -> not equal is false
        return {{0.0, 1.0}}; // otherwise could be mixed
    }
    return {{std::nullopt, std::nullopt}};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::tryInferMinMax(Operation *op) {
    if (auto inferMinMaxOp = llvm::dyn_cast<daphne::InferMinMax>(op))
        // If the operation implements the inference interface, we apply that.
        return inferMinMaxOp.inferMinMax();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<std::pair<std::optional<double>, std::optional<double>>> minmaxs;
        for (size_t i = 0; i < op->getNumResults(); i++)
            minmaxs.push_back({std::nullopt, std::nullopt});
        return minmaxs;
    }
}