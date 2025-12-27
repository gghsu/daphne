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

    struct MinMaxValues {
        Value minF64;
        Value maxF64;
    };

    void identifyMinMaxTargets(func::FuncOp f);
    std::optional<MinMaxValues> materializeMinMax(OpBuilder &builder, Location loc, Value matrix,
                                                  daphne::MatrixType mt);
    void processBlock(OpBuilder builder, Block *b);
    bool useSimdMinMax() const;

    const DaphneUserConfig &userConfig;
    llvm::DenseSet<Value> minMaxTargets;
};

void TransferDataPropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();
    minMaxTargets.clear();
    if (userConfig.use_vectorized_exec)
        identifyMinMaxTargets(f);

    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.getBody().front()));
}

void TransferDataPropertiesPass::identifyMinMaxTargets(func::FuncOp f) {
    f.walk([&](Operation *op) {
        if (!isa<mlir::daphne::EwGtOp, mlir::daphne::EwGeOp, mlir::daphne::EwLtOp, mlir::daphne::EwLeOp>(op))
            return;

        if (op->getNumOperands() < 1)
            return;

        Value matrixOperand = op->getOperand(0);
        auto mt = matrixOperand.getType().dyn_cast<daphne::MatrixType>();
        if (!mt)
            return;

        if (mt.getMinValue().has_value() && mt.getMaxValue().has_value())
            return;

        if (!matrixOperand.getDefiningOp())
            return;

        minMaxTargets.insert(matrixOperand);
    });
}

std::optional<TransferDataPropertiesPass::MinMaxValues>
TransferDataPropertiesPass::materializeMinMax(OpBuilder &builder, Location loc, Value matrix,
                                              daphne::MatrixType mt) {
    if (!userConfig.use_vectorized_exec)
        return std::nullopt;
    if (!minMaxTargets.contains(matrix))
        return std::nullopt;

    Type scalarTy = mt.getElementType();
    Value minScalar;
    Value maxScalar;

    if (useSimdMinMax()) {
        auto minOp = builder.create<mlir::daphne::MinAllSimdOp>(loc, scalarTy, matrix);
        auto maxOp = builder.create<mlir::daphne::MaxAllSimdOp>(loc, scalarTy, matrix);
        minOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
        maxOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
        minScalar = minOp.getResult();
        maxScalar = maxOp.getResult();
    } else {
        auto minOp = builder.create<mlir::daphne::AllAggMinOp>(loc, scalarTy, matrix);
        auto maxOp = builder.create<mlir::daphne::AllAggMaxOp>(loc, scalarTy, matrix);
        minOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
        maxOp->setAttr(CompilerUtils::ATTR_VEC, builder.getBoolAttr(true));
        minScalar = minOp.getResult();
        maxScalar = maxOp.getResult();
    }

    Type f64Ty = builder.getF64Type();
    Value minF64 = builder.create<mlir::daphne::CastOp>(loc, f64Ty, minScalar);
    Value maxF64 = builder.create<mlir::daphne::CastOp>(loc, f64Ty, maxScalar);
    return MinMaxValues{minF64, maxF64};
}

bool TransferDataPropertiesPass::useSimdMinMax() const {
    return userConfig.adaptiveAnalyze &&
           userConfig.adaptiveAnalyzeMode == DaphneUserConfig::AdaptiveAnalyzeMode::Simd;
}

void TransferDataPropertiesPass::processBlock(OpBuilder builder, Block *b) {
    for (Operation &op : b->getOperations()) {
        builder.setInsertionPointAfter(&op);
        Location loc = op.getLoc();
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

            auto minMaxValues = materializeMinMax(builder, loc, v, mt);

            Value coIsMinValue;
            Value coMinValue;
            if (mt.getMinValue().has_value()) {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMinValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMinValue().value()));
            } else if (minMaxValues) {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMinValue = minMaxValues->minF64;
            } else {
                coIsMinValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(false));
                coMinValue = builder.create<daphne::ConstantOp>(loc, -1.0);
            }

            Value coIsMaxValue;
            Value coMaxValue;
            if (mt.getMaxValue().has_value()) {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMaxValue = builder.create<daphne::ConstantOp>(loc, static_cast<double>(mt.getMaxValue().value()));
            } else if (minMaxValues) {
                coIsMaxValue = builder.create<daphne::ConstantOp>(loc, builder.getI1Type(), builder.getBoolAttr(true));
                coMaxValue = minMaxValues->maxF64;
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
