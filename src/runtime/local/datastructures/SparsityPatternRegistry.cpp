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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SparsityPatternRegistry.h"

// Initialize the static atomic counter to 11
// IDs 0-10 are reserved for special patterns (diagonal, identity, etc.)
// IDs >= 11 are for runtime-allocated patterns
std::atomic<size_t> SparsityPatternRegistry::nextPatternID{11};

