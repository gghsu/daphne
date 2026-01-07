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
#include <iomanip>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTResLhsPos, class DTLhsData, class DTRhsData> struct ColSemiJoin {
    static void apply(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData,
                      const DTRhsData *rhsData, int64_t numRes, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTResLhsPos, class DTLhsData, class DTRhsData>
void colSemiJoin(DTResLhsPos *&resLhsPos, const DTLhsData *lhsData, const DTRhsData *rhsData,
                 int64_t numRes, DCTX(ctx)) {
    ColSemiJoin<DTResLhsPos, DTLhsData, DTRhsData>::apply(resLhsPos, lhsData, rhsData, numRes, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename VTData, typename VTPos>
struct ColSemiJoin<Column<VTPos>, Column<VTData>, Column<VTData>> {
private:
    // Core semi-join logic using narrower key type
    template <typename VTKeyInternal>
    static size_t colSemiJoinCore(VTPos *valuesResLhsPos,
                                   const VTData *valuesLhsData, const VTData *valuesRhsData,
                                   size_t numLhsData, size_t numRhsData,
                                   VTData rhsMin, VTData rhsMax) {
        auto buildStart = std::chrono::steady_clock::now();
        
        // Build phase - use narrow key type for hash set
        absl::flat_hash_set<VTKeyInternal> hs;
        
        if constexpr (std::is_same_v<VTKeyInternal, VTData>) {
            // No offset transformation needed
            for (size_t r = 0; r < numRhsData; r++) {
                hs.insert(valuesRhsData[r]);
            }
            
            auto probeStart = std::chrono::steady_clock::now();
            double buildTime = std::chrono::duration<double>(probeStart - buildStart).count();

            size_t posRes = 0;
            size_t lookupCount = 0;
            for (size_t r = 0; r < numLhsData; r++) {
                lookupCount++;
                if (hs.contains(valuesLhsData[r])) {
                    valuesResLhsPos[posRes] = static_cast<VTPos>(r);
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
                hs.insert(key);
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
                    if (hs.contains(key)) {
                        valuesResLhsPos[posRes] = static_cast<VTPos>(r);
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
    static void apply(Column<VTPos> *&resLhsPos, const Column<VTData> *lhsData,
                      const Column<VTData> *rhsData, int64_t numRes, DCTX(ctx)) {
        
        auto startTime = std::chrono::steady_clock::now();

        const size_t numLhsData = lhsData->getNumRows();
        const size_t numRhsData = rhsData->getNumRows();
        
        if (numRes == -1)
            // Assuming FK-PK join.
            numRes = numLhsData;

        const VTData *valuesLhsData = lhsData->getValues();
        const VTData *valuesRhsData = rhsData->getValues();

        // Allocate result column
        if (resLhsPos == nullptr)
            resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);
        VTPos *valuesResLhsPos = resLhsPos->getValues();

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
                    posRes = colSemiJoinCore<uint8_t>(valuesResLhsPos, 
                                                       valuesLhsData, valuesRhsData, 
                                                       numLhsData, numRhsData, rhsMin, rhsMax);
                }
                // Dispatch to uint16_t keys if range fits
                else if (range <= 65535.0) {
                    keyTypePath = "uint16_t";
                    posRes = colSemiJoinCore<uint16_t>(valuesResLhsPos, 
                                                        valuesLhsData, valuesRhsData, 
                                                        numLhsData, numRhsData, rhsMin, rhsMax);
                }
                // Dispatch to uint32_t keys if range fits
                else if (range <= 4294967295.0) {
                    keyTypePath = "uint32_t";
                    posRes = colSemiJoinCore<uint32_t>(valuesResLhsPos, 
                                                        valuesLhsData, valuesRhsData, 
                                                        numLhsData, numRhsData, rhsMin, rhsMax);
                }
                else {
                    keyTypePath = "original-type";
                    // Use original VTData type
                    posRes = colSemiJoinCore<VTData>(valuesResLhsPos, 
                                                      valuesLhsData, valuesRhsData, 
                                                      numLhsData, numRhsData, rhsMin, rhsMax);
                }
            }
            else {
                keyTypePath = "no-minmax";
                // No min/max available, cannot use range optimization
                posRes = colSemiJoinCore<VTData>(valuesResLhsPos, 
                                                  valuesLhsData, valuesRhsData, 
                                                  numLhsData, numRhsData, 
                                                  static_cast<VTData>(0), static_cast<VTData>(0));
            }
        }
        else {
            keyTypePath = "non-arithmetic";
            // Non-arithmetic types
            posRes = colSemiJoinCore<VTData>(valuesResLhsPos, 
                                              valuesLhsData, valuesRhsData, 
                                              numLhsData, numRhsData, 
                                              VTData{}, VTData{});
        }

        resLhsPos->shrinkNumRows(posRes);

        auto endTime = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();
        std::cerr << "[KERNEL_TIME] ColSemiJoin: " << std::fixed << std::setprecision(6) 
                  << seconds << " seconds (Column " << numLhsData << "x" << numRhsData 
                  << ", key=" << keyTypePath << ")" << std::endl;
    }
};