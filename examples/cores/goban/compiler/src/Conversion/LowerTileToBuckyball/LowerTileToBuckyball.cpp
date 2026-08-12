//===- LowerTileToBuckyball.cpp - Goban tile pass registration ------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerTileToBuckyball/LowerTileToBuckyball.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
#include "Tile/Transform.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace {
class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<MemRefType>(op.getAMemArray().getType());
    auto outputType = dyn_cast<MemRefType>(op.getBMemArray().getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getRank() != 2 ||
        outputType.getRank() != 2)
      return op.emitError("requires static rank-2 input and output memrefs");
    if (outputType.getShape()[0] != inputType.getShape()[1] ||
        outputType.getShape()[1] != inputType.getShape()[0] ||
        inputType.getElementType() != outputType.getElementType())
      return op.emitError("requires a matching transposed output memref");

    rewriter.create<MemTransposeOp>(op.getLoc(), op.getAMemArray(),
                                    op.getBMemArray());
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)
  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert Goban Tile operations to Buckyball operations";
  }
  LowerTileToBuckyballPass() = default;
  LowerTileToBuckyballPass(const LowerTileToBuckyballPass &) {}

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Bank depth (rows per bank)."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Number of physical banks."),
                          llvm::cl::init(20)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tile::TileDialect, ::buddy::buckyball::BuckyballDialect,
                func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                scf::SCFDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addIllegalOp<tile::TileMatMulOp>();
    target.addIllegalOp<tile::TileTransposeOp>();

    RewritePatternSet patterns(context);
    mlir::buddy::populateMatrixTileMatMulPatterns(patterns, bankWidthBytes,
                                                  bankDepth, bankNum);
    patterns.add<TileTransposeLowering>(context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
