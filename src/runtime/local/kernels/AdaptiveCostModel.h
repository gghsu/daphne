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
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <random>

// Simple cost model for cache vs direct execution
namespace AdaptiveCostModel {
    struct CostParams {
        double Cf, CL, CI, A;
        CostParams() : Cf(1000.0), CL(10.0), CI(20.0), A(0.0) {}
        CostParams(double cf, double cl, double ci, double a = 0.0) 
            : Cf(cf), CL(cl), CI(ci), A(a) {}
    };
    
    inline size_t calculateThreshold(size_t n, const CostParams& params) {
        if (n == 0 || params.Cf + params.CI <= 0) {
            return 0;
        }
        
        double dvStar = (static_cast<double>(n) * (params.Cf - params.CL) - params.A) / (params.Cf + params.CI);
        return static_cast<size_t>(std::max(0.0, std::min(dvStar, static_cast<double>(n))));
    }
    
    inline double calculateThresholdRatio(size_t n, const CostParams& params) {
        if (n == 0) return 0.0;
        double dvStar = (static_cast<double>(n) * (params.Cf - params.CL) - params.A) / (params.Cf + params.CI);
        double ratio = dvStar / static_cast<double>(n);
        return std::max(0.0, std::min(ratio, 1.0));
    }
    
    inline bool shouldUseCache(size_t dv, size_t n, const CostParams& params) {
        return dv < calculateThreshold(n, params);
    }
    
    struct CostAnalysis {
        double defaultCost, cacheCost, savings;
        bool recommendCache;
        size_t threshold;
        double thresholdRatio;
    };
    
    inline CostAnalysis analyzeCosts(size_t dv, size_t n, const CostParams& params) {
        CostAnalysis analysis;
        analysis.defaultCost = static_cast<double>(n) * params.Cf;
        analysis.cacheCost = params.A + static_cast<double>(n) * params.CL + 
                           static_cast<double>(dv) * (params.CI + params.Cf);
        analysis.savings = analysis.defaultCost - analysis.cacheCost;
        analysis.threshold = calculateThreshold(n, params);
        analysis.thresholdRatio = (n > 0) ? static_cast<double>(analysis.threshold) / static_cast<double>(n) : 0.0;
        analysis.recommendCache = analysis.savings > 0.0;
        return analysis;
    }
    
    inline void logThresholdDecision(size_t dv, size_t threshold, bool actualDecision, double analysisTimeSeconds) {
        std::cerr << "[COST_MODEL] CostModelAnalysis dv=" << dv
                  << " threshold=" << threshold 
                  << " decision=" << (actualDecision ? "cache" : "default")
                  << " analysisTime=" << std::fixed << std::setprecision(6) << analysisTimeSeconds
                  << std::endl;
    }
        
    template<typename VTArg, typename VTRes>
    inline CostParams measureActualCosts(VTRes (*udf)(VTArg), const VTArg* actualData, size_t actualDataSize, 
                                       double currentAnalysisTimeSeconds = -1.0) {
        CostParams params;
        const size_t numSamples = std::min(actualDataSize, static_cast<size_t>(100));
        
        // Generate random sample indices
        std::vector<size_t> randomIndices;
        if (actualDataSize <= numSamples) {
            // Use all data if small enough
            for (size_t i = 0; i < actualDataSize; i++) {
                randomIndices.push_back(i);
            }
        } else {
            // Random sampling without replacement
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<size_t> dist(0, actualDataSize - 1);
            
            std::unordered_set<size_t> selected;
            while (selected.size() < numSamples) {
                selected.insert(dist(rng));
            }
            randomIndices.assign(selected.begin(), selected.end());
        }
        
        // Measure UDF cost on random samples
        auto startUdf = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < numSamples; i++) {
            volatile VTRes result = udf(actualData[randomIndices[i]]);
            (void)result;
        }
        auto endUdf = std::chrono::high_resolution_clock::now();
        params.Cf = std::chrono::duration<double, std::nano>(endUdf - startUdf).count() / numSamples;
        
        // Measure cache lookup cost
        std::unordered_map<VTArg, VTRes> lookupCache;
        for (size_t i = 0; i < numSamples / 2; i++) {
            size_t idx = randomIndices[i];
            lookupCache[actualData[idx]] = udf(actualData[idx]);
        }
        
        auto startLookup = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < numSamples; i++) {
            volatile auto it = lookupCache.find(actualData[randomIndices[i]]);
            (void)it;
        }
        auto endLookup = std::chrono::high_resolution_clock::now();
        params.CL = std::chrono::duration<double, std::nano>(endLookup - startLookup).count() / numSamples;
        
        // Measure cache insert cost
        std::unordered_map<VTArg, VTRes> insertCache;
        std::unordered_set<VTArg> uniqueVals;
        for (size_t i = 0; i < numSamples; i++) {
            uniqueVals.insert(actualData[randomIndices[i]]);
        }
        std::vector<VTArg> uniqueKeys(uniqueVals.begin(), uniqueVals.end());
        
        auto startInsert = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < uniqueKeys.size(); i++) {
            insertCache.emplace(uniqueKeys[i], udf(uniqueKeys[i]));
        }
        auto endInsert = std::chrono::high_resolution_clock::now();
        params.CI = std::chrono::duration<double, std::nano>(endInsert - startInsert).count() / uniqueKeys.size();

        // Set analysis cost
        params.A = (currentAnalysisTimeSeconds > 0.0) ? currentAnalysisTimeSeconds * 1e9 : 0.0;
        
        // Simple bounds checking
        params.Cf = std::max(1.0, params.Cf);
        params.CL = std::max(1.0, params.CL);
        params.CI = std::max(1.0, params.CI);
        
        return params;
    }
    
    } // namespace AdaptiveCostModel