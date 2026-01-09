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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <compiler/utils/CompilerUtils.h>
#include <api/cli/DaphneUserConfig.h>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseSet.h>

#include <optional>

using namespace mlir;

/**
 * @brief Inserts `analyzeData` kernel calls into the IR for adaptive property analysis based on adaptive_map.json.
 *
 * This pass walks through operations and their operands to inject runtime property analysis calls (analyzeData)
 * before operations that require specific data properties according to the adaptive_map configuration.
 * This enables adaptive optimization decisions at runtime based on actual data characteristics.
 */
struct AdaptiveAnalyzePropertiesPass : public PassWrapper<AdaptiveAnalyzePropertiesPass, OperationPass<func::FuncOp>> {
    explicit AdaptiveAnalyzePropertiesPass(const DaphneUserConfig &cfg, std::unordered_map<std::string, bool> &paths) 
        : userConfig(cfg), usedLibPaths(paths) {}

    void runOnOperation() final;
    StringRef getArgument() const final { return "adaptive-analyze-props"; }
    StringRef getDescription() const final { 
        return "Insert analyzeData calls for adaptive property analysis based on adaptive_map"; 
    }

    void processBlock(OpBuilder &builder, Block *b, Value dctx);

    const DaphneUserConfig &userConfig;
    std::unordered_map<std::string, bool> &usedLibPaths;
    llvm::DenseSet<Value> processedValues;  // Track which values already have properties analyzed
};

void AdaptiveAnalyzePropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();
    
    if (!userConfig.adaptiveAnalyze)
        return;
    
    processedValues.clear();
    OpBuilder builder(f.getContext());
    
    // Try to determine the DaphneContext
    // Skip functions that don't have a CreateDaphneContextOp (e.g., UDFs)
    Value dctx = nullptr;
    try {
        dctx = CompilerUtils::getDaphneContext(f);
    } catch (const std::exception &e) {
        // Function doesn't have DaphneContext (probably a UDF), skip it
        return;
    }
    
    processBlock(builder, &(f.getBody().front()), dctx);
}

void AdaptiveAnalyzePropertiesPass::processBlock(OpBuilder &builder, Block *b, Value dctx) {
    const KernelCatalog &kc = userConfig.kernelCatalog;
    
    for (Operation &op : llvm::make_early_inc_range(*b)) {
        Location loc = op.getLoc();
        const std::string opMnemonic = op.getName().stripDialect().str();
        
        // Check if this operation requires property analysis
        auto it = userConfig.adaptive_map.find(opMnemonic);
        if (it != userConfig.adaptive_map.end()) {
            const auto &props = it->second;
            
            // Collect all matrix/frame/column operands
            SmallVector<Value> inputsToAnalyze;
            for (Value operand : op.getOperands()) {
                Type operandType = operand.getType();
                if ((operandType.isa<mlir::daphne::MatrixType>() || 
                     operandType.isa<mlir::daphne::FrameType>() ||
                     operandType.isa<mlir::daphne::ColumnType>()) &&
                    !processedValues.contains(operand)) {
                    inputsToAnalyze.push_back(operand);
                }
            }

            // For each input, inject analyzeData call
            for (Value input : inputsToAnalyze) {
                processedValues.insert(input);
                
                // Determine which properties to analyze
                bool analyzeSparsity = false;
                bool analyzeSymmetric = false;
                bool analyzeSortness = false;
                bool analyzeMinMax = false;
                bool analyzeDistinct = false;
                bool analyzeSparsityPattern = false;
                
                for (const std::string &propertyName : props) {
                    if (propertyName == "sparsity") {
                        analyzeSparsity = true;
                    } else if (propertyName == "symmetric") {
                        analyzeSymmetric = true;
                    } else if (propertyName == "sortness" || propertyName == "sorted") {
                        analyzeSortness = true;
                    } else if (propertyName == "minmax" || propertyName == "min" || propertyName == "max") {
                        analyzeMinMax = true;
                    } else if (propertyName == "distinct" || propertyName == "numDistinct") {
                        analyzeDistinct = true;
                    } else if (propertyName == "sparsityPattern") {
                        analyzeSparsityPattern = true;
                    }
                }

                // Insert analyzeData call before this operation
                builder.setInsertionPoint(&op);
                
                // Create analyzeData operation
                builder.create<daphne::AnalyzeDataOp>(loc, input,
                    builder.create<daphne::ConstantOp>(loc, analyzeSparsity),
                    builder.create<daphne::ConstantOp>(loc, analyzeSymmetric),
                    builder.create<daphne::ConstantOp>(loc, analyzeSortness),
                    builder.create<daphne::ConstantOp>(loc, analyzeMinMax),
                    builder.create<daphne::ConstantOp>(loc, analyzeDistinct),
                    builder.create<daphne::ConstantOp>(loc, analyzeSparsityPattern));
            }
        }

        // Recurse into the op, if it has regions
        for (Region &r : op.getRegions())
            for (Block &b2 : r.getBlocks())
                processBlock(builder, &b2, dctx);
    }
}

std::unique_ptr<Pass> daphne::createAdaptiveAnalyzePropertiesPass(const DaphneUserConfig &cfg,
                                                                   std::unordered_map<std::string, bool> &usedLibPaths) {
    return std::make_unique<AdaptiveAnalyzePropertiesPass>(cfg, usedLibPaths);
}

// Version for auto-generated code (Passes.td)
std::unique_ptr<Pass> daphne::createAdaptiveAnalyzePropertiesPass() {
    static DaphneUserConfig defaultConfig;
    static std::unordered_map<std::string, bool> defaultLibPaths;
    return std::make_unique<AdaptiveAnalyzePropertiesPass>(defaultConfig, defaultLibPaths);
}
