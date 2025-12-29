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

#include <ir/daphneir/DataPropertyTypes.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/SparsityPatternRegistry.h>

#include <runtime/local/kernels/IsSymmetric.h>
#include <runtime/local/kernels/IsSorted.h>
#include <runtime/local/kernels/IsSortedSimd.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/NumDistinctCount.h>
#include <runtime/local/kernels/NumDistinctApprox.h>
#include <runtime/local/kernels/MinAllSimd.h>
#include <runtime/local/kernels/MaxAllSimd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <typeinfo>
#include <atomic>
#include <unordered_set>

namespace {
struct TPStats {
    size_t autoCount = 0;
    double autoTotal = 0.0;
    size_t flagManualCount = 0;
    double flagManualTotal = 0.0;
    size_t flagAdaptiveCount = 0;
    double flagAdaptiveTotal = 0.0;
    ~TPStats() {
        try {
            if(autoCount + flagManualCount + flagAdaptiveCount == 0) return;
            std::ofstream out("kernel_times.jsonl", std::ios::app);
            if(!out) return;
            const char *variant = std::getenv("DAPHNE_VARIANT");
            const char *iter = std::getenv("DAPHNE_ITER");
            auto emit = [&](const char *mode, size_t cnt, double tot) {
                if(cnt == 0) return;
                out << "{\"kernel\":\"TransferProperties\",\"mode\":\"" << mode
                    << "\",\"avgSeconds\":" << std::fixed << std::setprecision(6) << (tot / cnt)
                    << ",\"runs\":" << cnt << ",\"totalSeconds\":" << tot;
                if(variant) out << ",\"variant\":\"" << variant << "\"";
                if(iter) out << ",\"iter\":" << iter;
                out << "}" << '\n';
            };
            emit("automatic", autoCount, autoTotal);
            emit("flagged:manuallyAnalyze", flagManualCount, flagManualTotal);
            emit("flagged:adaptiveAnalyze", flagAdaptiveCount, flagAdaptiveTotal);
        } catch(...) { /* swallow */ }
    }
};
static TPStats __tpStats;
}

// Lightweight channel to expose the last measured analysis time (seconds) to other kernels (e.g., Map)
namespace TransferPropertiesRuntime {
    inline std::atomic<double> g_lastAnalyzeSeconds{0.0};
    inline void setLastAnalyzeSeconds(double secs) { g_lastAnalyzeSeconds.store(secs, std::memory_order_relaxed); }
    inline double getLastAnalyzeSeconds() { return g_lastAnalyzeSeconds.load(std::memory_order_relaxed); }
}

// Primary template declaration
template <class DT> struct TransferProperties {
    static void apply(const DT *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID, ssize_t sparsityPatternID,
                      DCTX(ctx), bool specificAnalysis  = false) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT> 
void transferProperties(const DT *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                        bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                        double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID, ssize_t sparsityPatternID,
                        DCTX(ctx)) {
    TransferProperties<DT>::apply(arg, is_sparsity, sparsity, is_symmetric, symmetric, is_sortness, sortness,
                                  is_minValue, minValue, is_maxValue, maxValue, is_distinct, distinct,
                                  is_sparsityPatternID, sparsityPatternID, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct TransferProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID, ssize_t sparsityPatternID,
                      DCTX(ctx), bool specificAnalysis  = false) {
        auto mat = const_cast<DenseMatrix<VT> *>(arg);

        // Step 1: Transfer compile-time known properties
        // Note: When specificAnalysis=true, is_* flags indicate "need to analyze", not "value is known"
        // So we only transfer properties when specificAnalysis=false (compiler pass providing known values)
        if(!specificAnalysis) {
            if(is_sparsity) {
                mat->is_sparsity = true;
                mat->sparsity = sparsity;
            }
            
            if(is_symmetric) {
                mat->is_symmetric = true;
                mat->symmetric = static_cast<BoolOrUnknown>(symmetric);
            }
            
            if(is_sortness) {
                mat->is_sortness = true;
                mat->sortness = static_cast<MatrixSortness>(sortness);
            }
            
            if(is_minValue) {
                mat->is_minValue = true;
                mat->minValue = minValue;
            }
            
            if(is_maxValue) {
                mat->is_maxValue = true;
                mat->maxValue = maxValue;
            }
            
            if(is_distinct) {
                mat->is_distinct = true;
                mat->distinct = distinct;
            }
            
            if(is_sparsityPatternID) {
                mat->is_sparsityPatternID = true;
                mat->sparsityPatternID = sparsityPatternID;
            }
        }

        // Step 2: Determine if runtime analysis is needed
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        const auto &userConfig = ctx->getUserConfig();

        // Determine analysis mode:
        // - automaticallyAnalyzeEverything: analyze all unknown properties with standard methods
        // - specificAnalysis: analyze only requested unknown properties (called via analyzeData)
        bool doAutoAnalysis = (!specificAnalysis) && userConfig.automaticallyAnalyzeEverything;
        bool doSpecificAnalysis = specificAnalysis;
        
        if(!doAutoAnalysis && !doSpecificAnalysis) {
            return;
        }

        // Step 3: Determine which properties need to be computed
        // - Auto mode: analyze all properties that are still unknown
        // - Specific mode: analyze only requested properties that are still unknown
        bool needSymmetric = !mat->is_symmetric && (doAutoAnalysis || (doSpecificAnalysis && is_symmetric));
        bool needSortness  = !mat->is_sortness  && (doAutoAnalysis || (doSpecificAnalysis && is_sortness));
        bool needMin       = !mat->is_minValue  && (doAutoAnalysis || (doSpecificAnalysis && is_minValue));
        bool needMax       = !mat->is_maxValue  && (doAutoAnalysis || (doSpecificAnalysis && is_maxValue));
        bool needDistinct  = !mat->is_distinct  && (doAutoAnalysis || (doSpecificAnalysis && is_distinct));
        bool needSparsityPattern = !mat->is_sparsityPatternID && (doAutoAnalysis || (doSpecificAnalysis && is_sparsityPatternID));

        bool analyzedAny = false;
        auto startTime = std::chrono::steady_clock::now();

        // Step 4: Perform runtime property analysis
        if constexpr(std::is_arithmetic_v<VT>) {
            if(needSymmetric) {
                mat->symmetric = isSymmetric(arg, ctx) ? BoolOrUnknown::True : BoolOrUnknown::False;
                mat->is_symmetric = true;
                analyzedAny = true;
            }
        }

        if(needSortness) {
            bool useSIMD = userConfig.adaptiveAnalyze && 
                          userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd;
            
            // Try SIMD version first if enabled
            if(useSIMD) {
                try {
                    int64_t sortResult = 0;
                    if constexpr(std::is_same_v<VT, double>) {
                        sortResult = isSortedSimd<DenseMatrix<double>>(
                            reinterpret_cast<const DenseMatrix<double> *>(arg), ctx);
                    }
                    else if constexpr(std::is_same_v<VT, float>) {
                        sortResult = isSortedSimd<DenseMatrix<float>>(
                            reinterpret_cast<const DenseMatrix<float> *>(arg), ctx);
                    }
                    mat->sortness = static_cast<MatrixSortness>(sortResult);
                    mat->is_sortness = true;
                    analyzedAny = true;
                } catch(const std::runtime_error &) {
                    // SIMD failed (e.g., wrong size), fall back to regular version
                    mat->sortness = static_cast<MatrixSortness>(isSorted(arg, ctx));
                    mat->is_sortness = true;
                    analyzedAny = true;
                }
            } else {
                // Use regular version
                mat->sortness = static_cast<MatrixSortness>(isSorted(arg, ctx));
                mat->is_sortness = true;
                analyzedAny = true;
            }
        }

        if constexpr(std::is_arithmetic_v<VT> && !std::is_same_v<VT, bool>) {
            if(needMin || needMax) {
                bool useSIMD = userConfig.adaptiveAnalyze && 
                              userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd;
                
                // Try SIMD version first if enabled
                if(useSIMD) {
                    try {
                        if(needMin) {
                            mat->minValue = minAllSimd<DenseMatrix<VT>, VT>(arg, ctx);
                            mat->is_minValue = true;
                            analyzedAny = true;
                        }
                        if(needMax) {
                            mat->maxValue = maxAllSimd<DenseMatrix<VT>, VT>(arg, ctx);
                            mat->is_maxValue = true;
                            analyzedAny = true;
                        }
                    } catch(const std::runtime_error &) {
                        // SIMD failed (e.g., wrong size), fall back to regular version
                        if(needMin) {
                            mat->minValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MIN, arg, ctx);
                            mat->is_minValue = true;
                            analyzedAny = true;
                        }
                        if(needMax) {
                            mat->maxValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MAX, arg, ctx);
                            mat->is_maxValue = true;
                            analyzedAny = true;
                        }
                    }
                } else {
                    // Use regular version
                    if(needMin) {
                        mat->minValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MIN, arg, ctx);
                        mat->is_minValue = true;
                        analyzedAny = true;
                    }
                    if(needMax) {
                        mat->maxValue = aggAll<VT, DenseMatrix<VT>>(AggOpCode::MAX, arg, ctx);
                        mat->is_maxValue = true;
                        analyzedAny = true;
                    }
                }
            }
        }

        if(needDistinct) {
            bool computed = false;
            auto analyzeMode = userConfig.adaptiveAnalyzeMode;
            
            // Option A: Approximate counting
            if(userConfig.adaptiveAnalyze && analyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Approx) {
                size_t K = userConfig.approxNDistinctK;
                int64_t seed = userConfig.approxSeed;
                size_t approxCount = numDistinctApprox<DenseMatrix<VT>>(arg, K, seed, ctx);
                mat->distinct = static_cast<ssize_t>(approxCount);
                mat->is_distinct = true;
                computed = true;
            }
            
            // Option B: Early-abort counting
            if(!computed && userConfig.adaptiveAnalyze && 
               analyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::EarlyAbort) {
                // Calculate threshold
                size_t threshold = 0;
                if(userConfig.mapDistinctThresholdIsRelative) {
                    double fraction = userConfig.mapDistinctThresholdFraction;
                    threshold = static_cast<size_t>(fraction * static_cast<double>(numRows * numCols));
                } else {
                    threshold = userConfig.mapDistinctThresholdAbsolute;
                }
                if(threshold == 0) threshold = 1;
                
                // Count distinct values with early abort
                size_t totalElements = numRows * numCols;
                std::unordered_set<VT> uniqueValues;
                uniqueValues.reserve(std::min(threshold, totalElements));
                
                const VT *values = arg->getValues();
                bool exceededThreshold = false;
                for(size_t i = 0; i < totalElements; ++i) {
                    uniqueValues.insert(values[i]);
                    if(uniqueValues.size() > threshold) {
                        exceededThreshold = true;
                        break;
                    }
                }
                
                if(exceededThreshold) {
                    mat->distinct = static_cast<ssize_t>(threshold + 1);
                } else {
                    mat->distinct = static_cast<ssize_t>(uniqueValues.size());
                }
                mat->is_distinct = true;
                computed = true;
            }
            
            // Option C: Exact counting
            if(!computed) {
                mat->distinct = numDistinctCount<DenseMatrix<VT>>(arg, ctx);
                mat->is_distinct = true;
            }
            analyzedAny = true;
        }

        // Note: sparsity pattern analysis not applicable for DenseMatrix

        // Exit early if nothing was analyzed
        if(!analyzedAny) return;

        // Record analysis time and statistics
        auto endTime = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(endTime - startTime).count();
        
        // Share analysis time with other kernels (e.g., Map cost model)
        TransferPropertiesRuntime::setLastAnalyzeSeconds(seconds);
        
        // Track statistics for different analysis modes
        bool isAdaptive = doSpecificAnalysis && userConfig.adaptiveAnalyze;
        if(doAutoAnalysis && !isAdaptive) {
            __tpStats.autoCount += 1;
            __tpStats.autoTotal += seconds;
        }
        else if(isAdaptive) {
            __tpStats.flagAdaptiveCount += 1;
            __tpStats.flagAdaptiveTotal += seconds;
        }

        // Log the analysis for debugging
        std::fprintf(stdout, "flaggedAnalysis: analyzed DenseMatrix<%s> [%zu x %zu] in %.6f s\n",
                     typeid(VT).name(), numRows, numCols, seconds);
        std::fflush(stdout);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct TransferProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, ssize_t minValue, bool is_maxValue,
                      ssize_t maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID, ssize_t sparsityPatternID,
                      DCTX(ctx), bool specificAnalysis  = false) {
        auto mat = const_cast<CSRMatrix<VT> *>(arg);

        // Step 1: Transfer compile-time known properties
        // Note: When specificAnalysis=true, is_* flags indicate "need to analyze", not "value is known"
        // So we only transfer properties when specificAnalysis=false (compiler pass providing known values)
        if(!specificAnalysis) {
            if(is_sparsity) {
                mat->is_sparsity = true;
                mat->sparsity = sparsity;
            }
            
            if(is_symmetric) {
                mat->is_symmetric = true;
                mat->symmetric = static_cast<BoolOrUnknown>(symmetric);
            }
            
            if(is_sortness) {
                mat->is_sortness = true;
                mat->sortness = static_cast<MatrixSortness>(sortness);
            }
            
            if(is_minValue) {
                mat->is_minValue = true;
                mat->minValue = minValue;
            }
            
            if(is_maxValue) {
                mat->is_maxValue = true;
                mat->maxValue = maxValue;
            }
            
            if(is_distinct) {
                mat->is_distinct = true;
                mat->distinct = distinct;
            }
                
            if (is_sparsityPatternID) {
                mat->is_sparsityPatternID = true;
                mat->sparsityPatternID = sparsityPatternID;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Column
// ----------------------------------------------------------------------------

template <typename VT>
struct TransferProperties<Column<VT>> {
    static void apply(const Column<VT> *arg, bool is_sparsity, double sparsity, bool is_symmetric, int64_t symmetric,
                      bool is_sortness, int64_t sortness, bool is_minValue, double minValue, bool is_maxValue,
                      double maxValue, bool is_distinct, ssize_t distinct, bool is_sparsityPatternID, ssize_t sparsityPatternID,
                      DCTX(ctx), bool specificAnalysis = false) {
        auto col = const_cast<Column<VT> *>(arg);
        
        // Check if we need to analyze properties at runtime
        const size_t numRows = arg->getNumRows();
        
        // Transfer compile-time properties to runtime column
        if(!specificAnalysis) {
            if(is_minValue && minValue != -1.0) {
                col->is_minValue = true;
                col->minValue = minValue;
            }
            
            if(is_maxValue && maxValue != -1.0) {
                col->is_maxValue = true;
                col->maxValue = maxValue;
            }
            
            if(is_sortness) {
                col->is_sortness = true;
                col->sortness = static_cast<MatrixSortness>(sortness);
            }
            
            if(is_distinct && distinct != -1) {
                col->is_distinct = true;
                col->distinct = distinct;
            }
        }
        
        const auto &userConfig = ctx->getUserConfig();
        
        bool doAutoAnalysis = (!specificAnalysis) && userConfig.automaticallyAnalyzeEverything;
        bool doSpecificAnalysis = specificAnalysis;
        if(!doAutoAnalysis && !doSpecificAnalysis) {
            return; // no analysis needed
        }
        
        // Check which properties need computation
        bool needSortness  = (doAutoAnalysis || (is_sortness  && !col->is_sortness));
        bool needMin       = (doAutoAnalysis || (is_minValue  && !col->is_minValue));
        bool needMax       = (doAutoAnalysis || (is_maxValue  && !col->is_maxValue));
        bool needDistinct  = (doAutoAnalysis || (is_distinct  && !col->is_distinct));
        
        bool analyzedAny = false;
        auto startTime = std::chrono::steady_clock::now();
        
        // Analyze min/max values for Column
        if constexpr(std::is_arithmetic_v<VT> && !std::is_same_v<VT, bool>) {
            if(needMin || needMax) {
                const VT *values = arg->getValues();
                if(numRows > 0) {
                    VT minVal = values[0];
                    VT maxVal = values[0];
                    
                    for(size_t i = 1; i < numRows; i++) {
                        if(values[i] < minVal) minVal = values[i];
                        if(values[i] > maxVal) maxVal = values[i];
                    }
                    
                    if(needMin) {
                        col->minValue = static_cast<double>(minVal);
                        col->is_minValue = true;
                        analyzedAny = true;
                    }
                    if(needMax) {
                        col->maxValue = static_cast<double>(maxVal);
                        col->is_maxValue = true;
                        analyzedAny = true;
                    }
                }
            }
        }
        
        // Analyze sortness for Column
        if(needSortness && numRows > 0) {
            const VT *values = arg->getValues();
            bool isAscending = true;
            bool isDescending = true;
            
            for(size_t i = 1; i < numRows && (isAscending || isDescending); i++) {
                if(values[i] < values[i-1]) isAscending = false;
                if(values[i] > values[i-1]) isDescending = false;
            }
            
            if(isAscending && isDescending)
                col->sortness = MatrixSortness::AllEqual;
            else if(isAscending)
                col->sortness = MatrixSortness::SortedAsc;
            else if(isDescending)
                col->sortness = MatrixSortness::SortedDesc;
            else
                col->sortness = MatrixSortness::NotSorted;
            
            col->is_sortness = true;
            analyzedAny = true;
        }
        
        // Analyze distinct count for Column
        if(needDistinct && numRows > 0) {
            const VT *values = arg->getValues();
            std::unordered_set<VT> uniqueValues;
            for(size_t i = 0; i < numRows; i++) {
                uniqueValues.insert(values[i]);
            }
            col->distinct = static_cast<ssize_t>(uniqueValues.size());
            col->is_distinct = true;
            analyzedAny = true;
        }
        
        if(!analyzedAny) return;
    }
};