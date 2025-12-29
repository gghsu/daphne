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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>

#include <absl/container/flat_hash_set.h>

#include <chrono>
#include <cstddef>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResLhsPos, class DTLhsData, class DTRhsData> struct ColSemiJoin {
    static void apply(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData, int64_t numRes,
                      DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResLhsPos, class DTLhsData, class DTRhsData>
void colSemiJoin(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData, int64_t numRes,
                 DCTX(ctx)) {
    ColSemiJoin<DTResLhsPos, DTLhsData, DTRhsData>::apply(resLhsPos, lhsData, rhsData, numRes, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos> struct ColSemiJoin<Column<VTPos>, Column<VTData>, Column<VTData>> {
    static void apply(Column<VTPos> *&resLhsPos, const Column<VTData> *lhsData, const Column<VTData> *rhsData,
                      int64_t numRes, DCTX(ctx)) {
        const size_t numLhsData = lhsData->getNumRows();
        const size_t numRhsData = rhsData->getNumRows();

        // Check if we should dispatch to a more efficient type
        if constexpr (std::is_same_v<VTPos, size_t> && std::is_arithmetic_v<VTData>) {
            // Only try to optimize if VTPos is size_t (the default/generic type) and VTData is numeric
            if (lhsData->is_minValue && lhsData->is_maxValue) {
                double range = lhsData->maxValue - lhsData->minValue;
                
                // Dispatch to uint32_t if range fits
                if (range <= 4294967295.0 && numLhsData <= 4294967295ULL) {
                    Column<uint32_t> *resLhsPos32 = nullptr;
                    ColSemiJoin<Column<uint32_t>, Column<VTData>, Column<VTData>>::apply(
                        resLhsPos32, lhsData, rhsData, numRes, ctx);
                                        
                    size_t numRows = resLhsPos32->getNumRows();
                    
                    // Convert result back to size_t
                    if (resLhsPos == nullptr)
                        resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    VTPos *valuesRes = resLhsPos->getValues();
                    const uint32_t *values32 = resLhsPos32->getValues();
                    for (size_t i = 0; i < numRows; i++)
                        valuesRes[i] = static_cast<VTPos>(values32[i]);
                    
                    DataObjectFactory::destroy(resLhsPos32);
                    return;
                }
                // Dispatch to uint8_t if range fits
                else if (range <= 255.0 && numLhsData <= 255ULL) {
                    Column<uint8_t> *resLhsPos8 = nullptr;
                    ColSemiJoin<Column<uint8_t>, Column<VTData>, Column<VTData>>::apply(
                        resLhsPos8, lhsData, rhsData, numRes, ctx);
                                        
                    size_t numRows = resLhsPos8->getNumRows();
                    
                    // Convert result back to size_t
                    if (resLhsPos == nullptr)
                        resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    VTPos *valuesRes = resLhsPos->getValues();
                    const uint8_t *values8 = resLhsPos8->getValues();
                    for (size_t i = 0; i < numRows; i++)
                        valuesRes[i] = static_cast<VTPos>(values8[i]);
                    
                    DataObjectFactory::destroy(resLhsPos8);
                    return;
                }
            }
        }
        
        if (numRes == -1)
            // Assuming FK-PK join.
            numRes = numLhsData;

        if (resLhsPos == nullptr)
            resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        VTPos *valuesResLhsPos = resLhsPos->getValues();

        // Build phase.
        absl::flat_hash_set<VTData> ht;
        const VTData *valuesRhsData = rhsData->getValues();
        for (size_t r = 0; r < numRhsData; r++)
            ht.insert(valuesRhsData[r]);

        // Probe phase.
        const VTData *valuesLhsData = lhsData->getValues();
        size_t posRes = 0;
        for (size_t r = 0; r < numLhsData; r++)
            if (ht.count(valuesLhsData[r]))
                valuesResLhsPos[posRes++] = r;

        resLhsPos->shrinkNumRows(posRes);
    }
};