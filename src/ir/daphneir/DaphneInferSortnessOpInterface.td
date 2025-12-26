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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSORTNESSOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERSORTNESSOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferSortnessOpInterface : OpInterface<"InferSortness"> {
    let description = [{
        Interface to infer whether the data object(s) returned by an operation are sortness.
    }];

    let methods = [
        InterfaceMethod<
                "Infer whether the output data object(s) are sortness.",
                "std::vector<MatrixSortness>", "inferSortness", (ins)
        >
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERSORTNESSOPINTERFACE_TD