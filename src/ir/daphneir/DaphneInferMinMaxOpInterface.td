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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERMINMAXOPINTERFACE_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERMINMAXOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def InferMinMaxOpInterface : OpInterface<"InferMinMax"> {
    let description = [{
        Interface to infer min and max value of the output data object(s).
    }];

    let methods = [
        InterfaceMethod<
                "Infer min and max value of the output data object(s)",
                "std::vector<std::pair<std::optional<double>, std::optional<double>>>", "inferMinMax", (ins)
        >
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERMINMAXOPINTERFACE_TD