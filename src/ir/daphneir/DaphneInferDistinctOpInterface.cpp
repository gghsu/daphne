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
#include <ir/daphneir/DaphneInferDistinctOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<ssize_t> daphne::SeqOp::inferDistinct() {
    Type fromTy = getFrom().getType();
    if (fromTy.isF64()) {
        try {
            double vFrom = CompilerUtils::constantOrThrow<double>(getFrom());
            double vTo = CompilerUtils::constantOrThrow<double>(getTo());
            double vInc = CompilerUtils::constantOrThrow<double>(getInc());
            return {static_cast<ssize_t>(floor(vTo / vInc - vFrom / vInc) + 1)};
        } catch (const std::runtime_error &e) {
            return {static_cast<ssize_t>(-1)};
        }
    }
    if (fromTy.isF32()) {
        try {
            float vFrom = CompilerUtils::constantOrThrow<float>(getFrom());
            float vTo = CompilerUtils::constantOrThrow<float>(getTo());
            float vInc = CompilerUtils::constantOrThrow<float>(getInc());
            return {static_cast<ssize_t>(floor(vTo / vInc - vFrom / vInc) + 1)};
        } catch (const std::runtime_error &e) {
            return {static_cast<ssize_t>(-1)};
        }
    } else if (fromTy.isSignedInteger(64)) {
        try {
            int64_t vFrom = CompilerUtils::constantOrThrow<int64_t>(getFrom());
            int64_t vTo = CompilerUtils::constantOrThrow<int64_t>(getTo());
            int64_t vInc = CompilerUtils::constantOrThrow<int64_t>(getInc());
            return {static_cast<ssize_t>(abs(vTo - vFrom) / abs(vInc) + 1)};
        } catch (const std::runtime_error &e) {
            return {static_cast<ssize_t>(-1)};
        }
    }
    throw ErrorHandler::compilerError(getLoc(), "InferDistinctOpInterface (daphne::SeqOp::inferDistinct)",
                                      "at the moment, shape inference for SeqOp supports only F64 and "
                                      "SI64 value types");
}


// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<ssize_t> daphne::tryInferDistinct(Operation *op) {
    if (auto inferDistinctOp = llvm::dyn_cast<daphne::InferDistinct>(op))
        // If the operation implements the inference interface, we apply that.
        return inferDistinctOp.inferDistinct();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<ssize_t> distincts;
        for (size_t i = 0; i < op->getNumResults(); i++)
            distincts.push_back(-1);
        return distincts;
    }
}