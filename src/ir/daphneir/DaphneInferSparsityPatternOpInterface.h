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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYPATTERNOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYPATTERNOPINTERFACE_H

#include "ir/daphneir/Daphne.h"

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSparsityPatternOpInterface.h.inc>
}

namespace mlir::daphne {
    /**
     * @brief Try to infer the sparsity pattern ID(s) of the result(s) of the given operation.
     *
     * If the operation implements the InferSparsityPattern interface, this function calls
     * the operation's inferSparsityPattern() method. Otherwise, it returns unknown pattern
     * IDs for all results.
     *
     * @param op The operation to infer sparsity pattern IDs for
     * @return A vector of pattern IDs, one for each result of the operation
     */
    std::vector<ssize_t> tryInferSparsityPattern(mlir::Operation *op);
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERSPARSITYPATTERNOPINTERFACE_H

