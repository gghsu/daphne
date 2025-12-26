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
#include <ir/daphneir/DaphneInferSparsityPatternOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;

// ****************************************************************************
// Compile-time pattern ID constants
// ****************************************************************************

// Special pattern IDs for certain sparsity patterns
// These IDs are assigned at compile-time and enable pattern equality detection
// without runtime comparison

namespace {
    // ****************************************************************************
    // Pattern ID Scheme:
    // - ID < 0: Unknown (needs runtime analysis)
    // - ID 0-10: Reserved special patterns (diagonal, identity, etc.)
    // - ID >= 11: Runtime-allocated patterns (from SparsityPatternRegistry)
    // ****************************************************************************

    // Unknown pattern (needs runtime analysis)
    constexpr ssize_t PATTERN_ID_UNKNOWN = -1;

    // Reserved special patterns (0-10)
    constexpr ssize_t PATTERN_ID_DIAGONAL = 0;
    constexpr ssize_t PATTERN_ID_IDENTITY = 1;
    constexpr ssize_t PATTERN_ID_TRIDIAGONAL = 2;
    constexpr ssize_t PATTERN_ID_UPPER_TRIANGULAR = 3;
    constexpr ssize_t PATTERN_ID_LOWER_TRIANGULAR = 4;
    // IDs 5-10 reserved for future special patterns

    // Runtime-allocated patterns start from 11
    constexpr ssize_t PATTERN_ID_RUNTIME_START = 11;
}

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<ssize_t> daphne::DiagMatrixOp::inferSparsityPattern() {
    return {PATTERN_ID_DIAGONAL};
}

// ****************************************************************************
// Unary operations: Pattern propagation
// ****************************************************************************

// Element-wise unary operations that preserve sparsity pattern
std::vector<ssize_t> daphne::EwSqrtOp::inferSparsityPattern() {
    auto inputMatrix = getArg().getType().dyn_cast<daphne::MatrixType>();
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};

    return {inputMatrix.getSparsityPatternID()};
}

std::vector<ssize_t> daphne::EwLnOp::inferSparsityPattern() {
    auto inputMatrix = getArg().getType().dyn_cast<daphne::MatrixType>();
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};

    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwAbsOp::inferSparsityPattern() {
    auto inputMatrix = getArg().getType().dyn_cast<daphne::MatrixType>();
    if (!inputMatrix)
        return {PATTERN_ID_UNKNOWN};
    
    return {inputMatrix.getSparsityPatternID()};
}

// ****************************************************************************
// Binary operations: Pattern combination
// ****************************************************************************

std::vector<ssize_t> daphne::EwAddOp::inferSparsityPattern() {
    auto lhsMatrix = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsMatrix = getRhs().getType().dyn_cast<daphne::MatrixType>();
    
    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};
    
    ssize_t lhsPatternID = lhsMatrix.getSparsityPatternID();
    ssize_t rhsPatternID = rhsMatrix.getSparsityPatternID();
    
    // If both operands have the same known pattern, result inherits it
    if (lhsPatternID >= 0 && lhsPatternID == rhsPatternID) {
        return {lhsPatternID};
    }
    
    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwSubOp::inferSparsityPattern() {
    auto lhsMatrix = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsMatrix = getRhs().getType().dyn_cast<daphne::MatrixType>();

    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};

    ssize_t lhsPatternID = lhsMatrix.getSparsityPatternID();
    ssize_t rhsPatternID = rhsMatrix.getSparsityPatternID();

    // If both operands have the same known pattern, result inherits it
    if (lhsPatternID >= 0 && lhsPatternID == rhsPatternID) {
        return {lhsPatternID};
    }

    return {PATTERN_ID_UNKNOWN};
}

std::vector<ssize_t> daphne::EwMulOp::inferSparsityPattern() {
    auto lhsMatrix = getLhs().getType().dyn_cast<daphne::MatrixType>();
    auto rhsMatrix = getRhs().getType().dyn_cast<daphne::MatrixType>();

    if (!lhsMatrix || !rhsMatrix)
        return {PATTERN_ID_UNKNOWN};

    ssize_t lhsPatternID = lhsMatrix.getSparsityPatternID();
    ssize_t rhsPatternID = rhsMatrix.getSparsityPatternID();

    // If both have the same pattern, result has the same pattern
    if (lhsPatternID >= 0 && lhsPatternID == rhsPatternID) {
        return {lhsPatternID};
    }

    return {PATTERN_ID_UNKNOWN};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<ssize_t> daphne::tryInferSparsityPattern(Operation *op) {
    if (auto inferPatternOp = llvm::dyn_cast<InferSparsityPattern>(op))
        // If the operation implements the inference interface, we apply that.
        return inferPatternOp.inferSparsityPattern();
    else {
        // If the operation does not implement the inference interface
        // we return unknown.
        std::vector<ssize_t> patternIDs;
        for (size_t i = 0; i < op->getNumResults(); i++)
            patternIDs.push_back(PATTERN_ID_UNKNOWN);
        return patternIDs;
    }
}

