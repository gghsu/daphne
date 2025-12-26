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

#pragma once

#include <atomic>
#include <cstddef>

// Manages unique IDs for sparsity patterns
// When two matrices have the same pattern ID, they share the same sparsity structure
class SparsityPatternRegistry {
private:
    static std::atomic<size_t> nextPatternID;
    
public:
    // Get a new unique pattern ID
    static size_t getNewID() {
        return nextPatternID.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Reset the counter to 0
    static void reset() {
        nextPatternID.store(0, std::memory_order_relaxed);
    }
    
    // Get the current max ID
    static size_t getCurrentMaxID() {
        return nextPatternID.load(std::memory_order_relaxed);
    }
};

