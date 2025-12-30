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
 * @brief Inserts `TransferPropertiesOp`s into the IR to ensure that compile-time information on the data properties of
 * DAPHNE data objects is transferred to the DAPHNE runtime.
 *
 * For each matrix-typed result of an operation, a `TransferPropertiesOp` is inserted that passes the compile-time
 * information on the matrix's data properties (extracted from the `MatrixType`) to the runtime. By inserting these
 * operations right after the creation of matrix-typed op results, we ensure that the runtime kernels subsequent
 * operations are lowered to have access to the property information. With this information available, the kernels can
 * do various runtime optimizations, e.g., choosing a more efficient algorithm on sorted or symmetric data.
 */
struct TransferDataPropertiesPass : public PassWrapper<TransferDataPropertiesPass, OperationPass<func::FuncOp>> {
    explicit TransferDataPropertiesPass(const DaphneUserConfig &cfg) : userConfig(cfg) {}

    void runOnOperation() final;
    StringRef getArgument() const final { return "transfer-data-props"; }
    StringRef getDescription() const final { return "TODO"; }

    void processBlock(OpBuilder builder, Block *b);

    const DaphneUserConfig &userConfig;
    llvm::DenseSet<Value> processedValues;  // Track which values already have min/max computed
};

void TransferDataPropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();
    processedValues.clear();
    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.getBody().front()));
}

void TransferDataPropertiesPass::processBlock(OpBuilder builder, Block *b) {
    for (Operation &op : b->getOperations()) {
        Location loc = op.getLoc();
        
        // STEP 1: Check if this operation enables vectorized execution and needs min/max for its inputs
        if (userConfig.use_vectorized_restricted && userConfig.adaptiveAnalyze) {
            const std::string opMnemonic = op.getName().stripDialect().str();
            auto it = userConfig.adaptive_map.find(opMnemonic);
            
            if (it != userConfig.adaptive_map.end()) {
                const auto &props = it->second;
                bool needsMinMax = std::find(props.begin(), props.end(), "minmax") != props.end();
                
                if (needsMinMax) {
                    // Check each matrix operand
                    for (Value operand : op.getOperands()) {
                        auto mt = operand.getType().dyn_cast<daphne::MatrixType>();
                        if (!mt || processedValues.contains(operand))
                            continue;
                        
                        // Skip if already has compile-time min/max
                        if (mt.getMinValue().has_value() && mt.getMaxValue().has_value())
                            continue;
                        
                        processedValues.insert(operand);
                        
                        // Insert min/max computation BEFORE this operation
                        builder.setInsertionPoint(&op);
                        Type scalarTy = mt.getElementType();
                        
                        Value minScalar, maxScalar;
                        if (userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd) {
                            auto minOp = builder.create<mlir::daphne::MinAllSimdOp>(loc, scalarTy, operand);
                            auto maxOp = builder.create<mlir::daphne::MaxAllSimdOp>(loc, scalarTy, operand);
                            minOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
                            maxOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
                            minScalar = minOp.getResult();
                            maxScalar = maxOp.getResult();
                        } else {
                            auto minOp = builder.create<mlir::daphne::AllAggMinOp>(loc, scalarTy, operand);
                            auto maxOp = builder.create<mlir::daphne::AllAggMaxOp>(loc, scalarTy, operand);
                            minOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
                            maxOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
                            minScalar = minOp.getResult();
                            maxScalar = maxOp.getResult();
                        }
                        
                        // Cast to f64 and attach to operand via TransferPropertiesOp
                        Value minF64 = builder.create<mlir::daphne::CastOp>(loc, builder.getF64Type(), minScalar);
                        Value maxF64 = builder.create<mlir::daphne::CastOp>(loc, builder.getF64Type(), maxScalar);
                        
                        auto coIsMin = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                        auto coIsMax = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                        
                        // Preserve other properties
                        bool isSparsity = mt.getSparsity() != -1.0;
                        auto coIsSparsity = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsity));
                        auto coSparsity = builder.create<daphne::ConstantOp>(loc, isSparsity ? mt.getSparsity() : -1.0);
                        
                        bool isSymmetric = mt.getSymmetric() != BoolOrUnknown::Unknown;
                        auto coIsSymmetric = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSymmetric));
                        auto coSymmetric = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSymmetric()));
                        
                        bool isSortness = mt.getSortness() != MatrixSortness::Unknown;
                        auto coIsSortness = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSortness));
                        auto coSortness = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSortness()));
                        
                        bool isDistinct = mt.getDistinct() != -1;
                        auto coIsDistinct = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isDistinct));
                        auto coDistinct = builder.create<daphne::ConstantOp>(loc, isDistinct ? mt.getDistinct() : -1);
                        
                        bool isSparsityPatternID = mt.getSparsityPatternID() != -1;
                        auto coIsSparsityPatternID = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsityPatternID));
                        auto coSparsityPatternID = builder.create<daphne::ConstantOp>(loc, isSparsityPatternID ? mt.getSparsityPatternID() : -1);
                        
                        builder.create<daphne::TransferPropertiesOp>(loc, operand, 
                            coIsSparsity, coSparsity, coIsSymmetric, coSymmetric,
                            coIsSortness, coSortness, coIsMin, minF64, coIsMax, maxF64,
                            coIsDistinct, coDistinct, coIsSparsityPatternID, coSparsityPatternID);
                    }
                }
            }
        }
        
        // STEP 2: Add TransferPropertiesOp for all operation results (standard behavior)
        builder.setInsertionPointAfter(&op);
        
        for (Value v : op.getResults()) {
            Type t = v.getType();
            auto mt = t.dyn_cast<daphne::MatrixType>();
            if (!mt)
                continue;

            bool isSparsity = mt.getSparsity() != -1.0;
            auto coIsSparsity = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsity));
            auto coSparsity = builder.create<daphne::ConstantOp>(loc, isSparsity ? mt.getSparsity() : -1.0);

            bool isSymmetric = mt.getSymmetric() != BoolOrUnknown::Unknown;
            auto coIsSymmetric = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSymmetric));
            auto coSymmetric = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSymmetric()));

            bool isSortness = mt.getSortness() != MatrixSortness::Unknown;
            auto coIsSortness = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSortness));
            auto coSortness = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSortness()));

            Value coIsMinValue;
            Value coMinValue;
            if (mt.getMinValue().has_value()) {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMinValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMinValue().value()));
            } else {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                coMinValue = builder.create<daphne::ConstantOp>(loc, -1.0);
            }

            Value coIsMaxValue;
            Value coMaxValue;
            if (mt.getMaxValue().has_value()) {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMaxValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMaxValue().value()));
            } else {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                coMaxValue = builder.create<daphne::ConstantOp>(loc, -1.0);
            }

            bool isDistinct = mt.getDistinct() != -1;
            auto coIsDistinct = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isDistinct));
            auto coDistinct = builder.create<daphne::ConstantOp>(loc, isDistinct ? mt.getDistinct() : -1);

            bool isSparsityPatternID = mt.getSparsityPatternID() != -1;
            auto coIsSparsityPatternID = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(isSparsityPatternID));
            auto coSparsityPatternID = builder.create<daphne::ConstantOp>(loc,
                                                                           isSparsityPatternID ? mt.getSparsityPatternID() : -1);

            builder.create<daphne::TransferPropertiesOp>(loc, v, coIsSparsity, coSparsity, coIsSymmetric, coSymmetric,
                                                         coIsSortness, coSortness, coIsMinValue, coMinValue, coIsMaxValue,
                                                         coMaxValue, coIsDistinct, coDistinct, coIsSparsityPatternID,
                                                         coSparsityPatternID);
        }

        // Recurse into the op, if it has regions.
        for (Region &r : op.getRegions())
            for (Block &b2 : r.getBlocks())
                processBlock(builder, &b2);
    }
}

std::unique_ptr<Pass> daphne::createTransferDataPropertiesPass() {
    static DaphneUserConfig defaultCfg;
    return std::make_unique<TransferDataPropertiesPass>(defaultCfg);
}

std::unique_ptr<Pass> daphne::createTransferDataPropertiesPass(const DaphneUserConfig &cfg) {
    return std::make_unique<TransferDataPropertiesPass>(cfg);
}
