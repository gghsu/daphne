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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AdaptiveCostModel.h>
#include <runtime/local/kernels/TransferProperties.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <typeinfo>
#include <iomanip>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg> struct Map {
    // We could have a more specialized function pointer here i.e.
    // (DTRes::VT)(*func)(DTArg::VT). The problem is that this is currently not
    // supported by kernels.json.
    static void apply(DTRes *&res, const DTArg *arg, void *func, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg> void map(DTRes *&res, const DTArg *arg, void *func, DCTX(ctx)) {
    Map<DTRes, DTArg>::apply(res, arg, func, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct Map<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {
    static void apply(DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, void *func, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        auto udf = reinterpret_cast<VTRes (*)(VTArg)>(func);

        const VTArg *valuesArg = arg->getValues();
        VTRes *valuesRes = res->getValues();
       
        const ssize_t argDistinct = arg->distinct;
        const size_t n = numRows * numCols;

        // Determine threshold: adaptive cost model or static configuration
        size_t distinctThreshold = 1;
        bool useAdaptiveThreshold = true;
        
        // Read user configuration
        if(ctx != nullptr && argDistinct > 0) {
            const auto &uc = ctx->getUserConfig();
            useAdaptiveThreshold = uc.mapDistinctThresholdAdaptive;
            std::cerr << "[DEBUG] Map.h: useAdaptiveThreshold=" << useAdaptiveThreshold << std::endl;
            
            // Static threshold configuration (only if not using adaptive)
            if(!useAdaptiveThreshold) {
                if(uc.mapDistinctThresholdIsRelative) {
                    double fraction = uc.mapDistinctThresholdFraction;
                    distinctThreshold = static_cast<size_t>(n * fraction);
                } else {
                    distinctThreshold = uc.mapDistinctThresholdAbsolute;
                }
                if(distinctThreshold == 0) distinctThreshold = 1;
            }
        } else {
            // Default fallback: 120% of total size (effectively disabled)
            distinctThreshold = static_cast<size_t>(n * 1.2);
        }
        
        // Adaptive cost model analysis (overrides static threshold)
        if(argDistinct > 0 && useAdaptiveThreshold) {
            auto costModelStart = std::chrono::high_resolution_clock::now();
            
            // Measure actual costs (Cf, CL, CI) via micro-benchmarking
            const double tp_last = TransferPropertiesRuntime::getLastAnalyzeSeconds();
            auto measuredParams = AdaptiveCostModel::measureActualCosts(udf, valuesArg, n, tp_last > 0.0 ? tp_last : -1.0);
            
            // Calculate optimal threshold and override
            size_t adaptiveThr = AdaptiveCostModel::calculateThreshold(n, measuredParams);
            if(adaptiveThr == 0) adaptiveThr = 1;
            distinctThreshold = adaptiveThr;
            
            auto costModelEnd = std::chrono::high_resolution_clock::now();
            double costModelTime = std::chrono::duration<double>(costModelEnd - costModelStart).count();
            
            bool actualDecision = (static_cast<size_t>(argDistinct) <= distinctThreshold);
            AdaptiveCostModel::logThresholdDecision(
                static_cast<size_t>(argDistinct), adaptiveThr, actualDecision, costModelTime);
        }

        // Start timing the kernel's core execution
        auto map_start = std::chrono::high_resolution_clock::now();

        bool usedCache = false;
        if (argDistinct > 0 && static_cast<size_t>(argDistinct) <= distinctThreshold) {
            std::unordered_map<VTArg, VTRes> cache;
            cache.reserve(static_cast<size_t>(argDistinct));
            const size_t n = numRows * numCols;
            for (size_t i = 0; i < n; ++i) {
                VTArg val = valuesArg[i];
                auto it = cache.find(val);
                if (it == cache.end()) {
                    VTRes outVal = udf(val);
                    it = cache.emplace(val, outVal).first;
                }
                valuesRes[i] = it->second;
            }
            usedCache = true;
        } else {
            // Fallback: standard map
            const size_t n = numRows * numCols;
            for (size_t i = 0; i < n; ++i) {
                valuesRes[i] = udf(valuesArg[i]);
            }
        }

        auto map_end = std::chrono::high_resolution_clock::now();
        double map_secs = std::chrono::duration<double>(map_end - map_start).count();
        
        std::cerr << "[KERNEL_TIME] Map: " << std::fixed << std::setprecision(6) 
                  << map_secs << " seconds (DenseMatrix " << numRows << "x" << numCols 
                  << ", mode=" << (usedCache ? "cache" : "default") << ")" << std::endl;    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg> struct Map<Matrix<VTRes>, Matrix<VTArg>> {
    static void apply(Matrix<VTRes> *&res, const Matrix<VTArg> *arg, void *func, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, numCols, false);

        auto udf = reinterpret_cast<VTRes (*)(VTArg)>(func);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, udf(arg->get(r, c)));
        res->finishAppend();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_MAP_H
