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
#include <iomanip>

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
private:
    // Core hash join logic using narrower key type
    template <typename VTKeyInternal>
    static size_t colJoinCore(VTPos *valuesResLhsPos, VTPos *valuesResRhsPos,
                              const VTData *valuesLhsData, const VTData *valuesRhsData,
                              size_t numLhsData, size_t numRhsData,
                              VTData rhsMin, VTData rhsMax) {
        auto buildStart = std::chrono::steady_clock::now();
        
        // Build phase - use narrow key type for hash table
        absl::flat_hash_map<VTKeyInternal, VTPos> ht;
        
        if constexpr (std::is_same_v<VTKeyInternal, VTData>) {
            // No offset transformation needed
            for (size_t r = 0; r < numRhsData; r++) {
                ht[valuesRhsData[r]] = static_cast<VTPos>(r);
            }
            
            auto probeStart = std::chrono::steady_clock::now();
            double buildTime = std::chrono::duration<double>(probeStart - buildStart).count();

            size_t posRes = 0;
            size_t lookupCount = 0;
            for (size_t r = 0; r < numLhsData; r++) {
                lookupCount++;
                auto it = ht.find(valuesLhsData[r]);
                if (it != ht.end()) {
                    valuesResLhsPos[posRes] = static_cast<VTPos>(r);
                    valuesResRhsPos[posRes] = it->second;
                    posRes++;
                }
            }
            
            auto probeEnd = std::chrono::steady_clock::now();
            double probeTime = std::chrono::duration<double>(probeEnd - probeStart).count();
            
            return posRes;
        } else {
            // Use offset transformation for narrower key types
            for (size_t r = 0; r < numRhsData; r++) {
                VTKeyInternal key = static_cast<VTKeyInternal>(valuesRhsData[r] - rhsMin);
                ht[key] = static_cast<VTPos>(r);
            }
            
            auto probeStart = std::chrono::steady_clock::now();
            double buildTime = std::chrono::duration<double>(probeStart - buildStart).count();

            // Probe phase - convert keys on-the-fly
            size_t posRes = 0;
            size_t rangeCheckCount = 0;
            size_t lookupCount = 0;
            for (size_t r = 0; r < numLhsData; r++) {
                VTData lhsVal = valuesLhsData[r];
                rangeCheckCount++;
                // Range check: skip values outside rhs range
                if (lhsVal >= rhsMin && lhsVal <= rhsMax) {
                    lookupCount++;
                    VTKeyInternal key = static_cast<VTKeyInternal>(lhsVal - rhsMin);
                    auto it = ht.find(key);
                    if (it != ht.end()) {
                        valuesResLhsPos[posRes] = static_cast<VTPos>(r);
                        valuesResRhsPos[posRes] = it->second;
                        posRes++;
                    }
                }
            }
            
            auto probeEnd = std::chrono::steady_clock::now();
            double probeTime = std::chrono::duration<double>(probeEnd - probeStart).count();
            double filterRate = (rangeCheckCount - lookupCount) * 100.0 / rangeCheckCount;
            std::cerr << "  [filtered " << std::fixed << std::setprecision(1) 
                      << filterRate << "% lookups]" << std::endl;
            
            return posRes;
        }
    }

public:
    static void apply(Column<VTPos> *&resLhsPos, Column<VTPos> *&resRhsPos, const Column<VTData> *lhsData,
                      const Column<VTData> *rhsData, int64_t numRes, DCTX(ctx)) {
        
        auto startTime = std::chrono::steady_clock::now();

        const size_t numLhsData = lhsData->getNumRows();
        const size_t numRhsData = rhsData->getNumRows();
        
        if (numRes == -1)
            // Assuming FK-PK join.
            numRes = numLhsData;

        const VTData *valuesLhsData = lhsData->getValues();
        const VTData *valuesRhsData = rhsData->getValues();

        // Allocate result columns
        if (resLhsPos == nullptr)
            resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        if (resRhsPos == nullptr)
            resRhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        VTPos *valuesResLhsPos = resLhsPos->getValues();
        VTPos *valuesResRhsPos = resRhsPos->getValues();

        size_t posRes = 0;

        // Check if we should dispatch to a narrower key type based on rhs data range
        std::string keyTypePath;
        if constexpr (std::is_arithmetic_v<VTData>) {
            if (rhsData->is_minValue && rhsData->is_maxValue) {
                VTData rhsMin = static_cast<VTData>(rhsData->minValue);
                VTData rhsMax = static_cast<VTData>(rhsData->maxValue);
                double range = rhsData->maxValue - rhsData->minValue;
                
                // Dispatch to uint8_t keys if range fits
                if (range <= 255.0) {
                    keyTypePath = "uint8_t";
                    posRes = colJoinCore<uint8_t>(valuesResLhsPos, valuesResRhsPos, 
                                                   valuesLhsData, valuesRhsData, 
                                                   numLhsData, numRhsData, rhsMin, rhsMax);
                }
                // Dispatch to uint16_t keys if range fits
                else if (range <= 65535.0) {
                    keyTypePath = "uint16_t";
                    posRes = colJoinCore<uint16_t>(valuesResLhsPos, valuesResRhsPos, 
                                                    valuesLhsData, valuesRhsData, 
                                                    numLhsData, numRhsData, rhsMin, rhsMax);
                }
                // Dispatch to uint32_t keys if range fits
                else if (range <= 4294967295.0) {
                    keyTypePath = "uint32_t";
                    posRes = colJoinCore<uint32_t>(valuesResLhsPos, valuesResRhsPos, 
                                                    valuesLhsData, valuesRhsData, 
                                                    numLhsData, numRhsData, rhsMin, rhsMax);
                }
                else {
                    keyTypePath = "original-type";
                    // Use original VTData type
                    posRes = colJoinCore<VTData>(valuesResLhsPos, valuesResRhsPos, 
                                                  valuesLhsData, valuesRhsData, 
                                                  numLhsData, numRhsData, rhsMin, rhsMax);
                }
            }
            else {
                keyTypePath = "no-minmax";
                // No min/max available, cannot use range optimization
                posRes = colJoinCore<VTData>(valuesResLhsPos, valuesResRhsPos, 
                                              valuesLhsData, valuesRhsData, 
                                              numLhsData, numRhsData, 
                                              static_cast<VTData>(0), static_cast<VTData>(0));
            }
        }
        else {
            keyTypePath = "non-arithmetic";
            // Non-arithmetic types
            posRes = colJoinCore<VTData>(valuesResLhsPos, valuesResRhsPos, 
                                          valuesLhsData, valuesRhsData, 
                                          numLhsData, numRhsData, 
                                          VTData{}, VTData{});
        }

        resLhsPos->shrinkNumRows(posRes);
        resRhsPos->shrinkNumRows(posRes);

        auto endTime = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();
        std::cerr << "[KERNEL_TIME] ColJoin: " << std::fixed << std::setprecision(6) 
                  << seconds << " seconds (Column " << numLhsData << "x" << numRhsData 
                  << ", key=" << keyTypePath << ")" << std::endl;
    }
};