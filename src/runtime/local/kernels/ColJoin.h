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

#include <absl/container/flat_hash_map.h>

#include <chrono>
#include <cstddef>
#include <iostream>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResLhsPos, class DTResRhsPos, class DTLhsData, class DTRhsData> struct ColJoin {
    static void apply(DTResLhsPos *&resLhsPos, DTResRhsPos *&resRhsPos, const DTLhsData *lhsData,
                      const DTRhsData *rhsData, int64_t numRes, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResLhsPos, class DTResRhsPos, class DTLhsData, class DTRhsData>
void colJoin(DTResLhsPos *&resLhsPos, DTResRhsPos *&resRhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData,
             int64_t numRes, DCTX(ctx)) {
    ColJoin<DTResLhsPos, DTResRhsPos, DTLhsData, DTRhsData>::apply(resLhsPos, resRhsPos, lhsData, rhsData, numRes, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column, Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos>
struct ColJoin<Column<VTPos>, Column<VTPos>, Column<VTData>, Column<VTData>> {
    static void apply(Column<VTPos> *&resLhsPos, Column<VTPos> *&resRhsPos, const Column<VTData> *lhsData,
                      const Column<VTData> *rhsData, int64_t numRes, DCTX(ctx)) {
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
                    Column<uint32_t> *resRhsPos32 = nullptr;
                    ColJoin<Column<uint32_t>, Column<uint32_t>, Column<VTData>, Column<VTData>>::apply(
                        resLhsPos32, resRhsPos32, lhsData, rhsData, numRes, ctx);
                                        
                    size_t numRows = resLhsPos32->getNumRows();
                    
                    // Convert result back to size_t
                    if (resLhsPos == nullptr)
                        resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    if (resRhsPos == nullptr)
                        resRhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    VTPos *valuesResLhs = resLhsPos->getValues();
                    VTPos *valuesResRhs = resRhsPos->getValues();
                    const uint32_t *values32Lhs = resLhsPos32->getValues();
                    const uint32_t *values32Rhs = resRhsPos32->getValues();
                    for (size_t i = 0; i < numRows; i++) {
                        valuesResLhs[i] = static_cast<VTPos>(values32Lhs[i]);
                        valuesResRhs[i] = static_cast<VTPos>(values32Rhs[i]);
                    }
                    
                    DataObjectFactory::destroy(resLhsPos32);
                    DataObjectFactory::destroy(resRhsPos32);
                    return;
                }
                // Dispatch to uint8_t if range fits
                else if (range <= 255.0 && numLhsData <= 255ULL) {
                    Column<uint8_t> *resLhsPos8 = nullptr;
                    Column<uint8_t> *resRhsPos8 = nullptr;
                    ColJoin<Column<uint8_t>, Column<uint8_t>, Column<VTData>, Column<VTData>>::apply(
                        resLhsPos8, resRhsPos8, lhsData, rhsData, numRes, ctx);
                                        
                    size_t numRows = resLhsPos8->getNumRows();
                    
                    // Convert result back to size_t
                    if (resLhsPos == nullptr)
                        resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    if (resRhsPos == nullptr)
                        resRhsPos = DataObjectFactory::create<Column<VTPos>>(numRows, false);
                    VTPos *valuesResLhs = resLhsPos->getValues();
                    VTPos *valuesResRhs = resRhsPos->getValues();
                    const uint8_t *values8Lhs = resLhsPos8->getValues();
                    const uint8_t *values8Rhs = resRhsPos8->getValues();
                    for (size_t i = 0; i < numRows; i++) {
                        valuesResLhs[i] = static_cast<VTPos>(values8Lhs[i]);
                        valuesResRhs[i] = static_cast<VTPos>(values8Rhs[i]);
                    }
                    
                    DataObjectFactory::destroy(resLhsPos8);
                    DataObjectFactory::destroy(resRhsPos8);
                    return;
                }
            }
        }    
        if (numRes == -1)
            // Assuming FK-PK join.
            numRes = numLhsData;

        if (resLhsPos == nullptr)
            resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        if (resRhsPos == nullptr)
            resRhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        VTPos *valuesResLhsPos = resLhsPos->getValues();
        VTPos *valuesResRhsPos = resRhsPos->getValues();

        // Build phase.
        absl::flat_hash_map<VTData, VTPos> ht;
        const VTData *valuesRhsData = rhsData->getValues();
        for (size_t r = 0; r < numRhsData; r++)
            ht[valuesRhsData[r]] = r;

        // Probe phase.
        const VTData *valuesLhsData = lhsData->getValues();
        size_t posRes = 0;
        for (size_t r = 0; r < numLhsData; r++) {
            auto it = ht.find(valuesLhsData[r]);
            if (it != ht.end()) {
                valuesResLhsPos[posRes] = r;
                valuesResRhsPos[posRes] = it->second;
                posRes++;
            }
        }

        resLhsPos->shrinkNumRows(posRes);
        resRhsPos->shrinkNumRows(posRes);
    }
};