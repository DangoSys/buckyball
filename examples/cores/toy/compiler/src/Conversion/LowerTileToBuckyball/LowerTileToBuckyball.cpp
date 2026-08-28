//===- LowerTileToBuckyball.cpp - Toy tile->buckyball pass ---------------===//
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

#include "Conversion/LowerTileToBuckyball/LowerTileToBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
#include "Tile/Transform.h"

using namespace mlir;
namespace tile = ::buddy::tile;
using mlir::buddy::kDefaultBankWidthBytes;

namespace {

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  explicit TileConv2dLowering(MLIRContext *context, int64_t bankWidthBytes,
                              int64_t bankDepth, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes),
        bankDepth(bankDepth) {}

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();
    Value filter = op.getFilter();
    Value output = op.getOutput();

    auto inType = cast<MemRefType>(input.getType());
    auto filterType = cast<MemRefType>(filter.getType());
    auto outType = cast<MemRefType>(output.getType());

    auto inShape = inType.getShape();
    auto fShape = filterType.getShape();
    auto outShape = outType.getShape();

    int64_t N = inShape[0], H = inShape[1], W = inShape[2], C = inShape[3];
    int64_t KH = fShape[0], KW = fShape[1], OC = fShape[3];
    int64_t OH = outShape[1], OW = outShape[2];

    if (!inType.getElementType().isF32() ||
        !filterType.getElementType().isF32())
      return op.emitError("tile_conv2d lowering currently expects f32");
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0 || KH <= 0 || KW <= 0 || OC <= 0 ||
        OH <= 0 || OW <= 0)
      return op.emitError("tile_conv2d requires positive static shapes");
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value nUpper = rewriter.create<arith::ConstantIndexOp>(loc, N);
    Value ohUpper = rewriter.create<arith::ConstantIndexOp>(loc, OH);
    Value owUpper = rewriter.create<arith::ConstantIndexOp>(loc, OW);
    Value ocUpper = rewriter.create<arith::ConstantIndexOp>(loc, OC);
    Value khUpper = rewriter.create<arith::ConstantIndexOp>(loc, KH);
    Value kwUpper = rewriter.create<arith::ConstantIndexOp>(loc, KW);
    Value cUpper = rewriter.create<arith::ConstantIndexOp>(loc, C);
    Value zeroF32 = cstF32(rewriter, loc, 0.0f);

    auto nLoop = rewriter.create<scf::ForOp>(loc, zero, nUpper, one);
    {
      OpBuilder::InsertionGuard nGuard(rewriter);
      rewriter.setInsertionPointToStart(nLoop.getBody());
      Value nIv = nLoop.getInductionVar();

      auto ohLoop = rewriter.create<scf::ForOp>(loc, zero, ohUpper, one);
      {
        OpBuilder::InsertionGuard ohGuard(rewriter);
        rewriter.setInsertionPointToStart(ohLoop.getBody());
        Value ohIv = ohLoop.getInductionVar();

        auto owLoop = rewriter.create<scf::ForOp>(loc, zero, owUpper, one);
        {
          OpBuilder::InsertionGuard owGuard(rewriter);
          rewriter.setInsertionPointToStart(owLoop.getBody());
          Value owIv = owLoop.getInductionVar();

          auto ocLoop = rewriter.create<scf::ForOp>(loc, zero, ocUpper, one);
          {
            OpBuilder::InsertionGuard ocGuard(rewriter);
            rewriter.setInsertionPointToStart(ocLoop.getBody());
            Value ocIv = ocLoop.getInductionVar();

            Value init = zeroF32;
            auto khLoop =
                rewriter.create<scf::ForOp>(loc, zero, khUpper, one, init);
            {
              OpBuilder::InsertionGuard khGuard(rewriter);
              rewriter.setInsertionPointToStart(khLoop.getBody());
              Value khIv = khLoop.getInductionVar();
              Value khAcc = khLoop.getRegionIterArgs().front();

              auto kwLoop =
                  rewriter.create<scf::ForOp>(loc, zero, kwUpper, one, khAcc);
              {
                OpBuilder::InsertionGuard kwGuard(rewriter);
                rewriter.setInsertionPointToStart(kwLoop.getBody());
                Value kwIv = kwLoop.getInductionVar();
                Value kwAcc = kwLoop.getRegionIterArgs().front();

                auto cLoop =
                    rewriter.create<scf::ForOp>(loc, zero, cUpper, one, kwAcc);
                {
                  OpBuilder::InsertionGuard cGuard(rewriter);
                  rewriter.setInsertionPointToStart(cLoop.getBody());
                  Value cIv = cLoop.getInductionVar();
                  Value cAcc = cLoop.getRegionIterArgs().front();
                  Value inH = rewriter.create<arith::AddIOp>(loc, ohIv, khIv);
                  Value inW = rewriter.create<arith::AddIOp>(loc, owIv, kwIv);
                  Value inValue = rewriter.create<memref::LoadOp>(
                      loc, input, ValueRange{nIv, inH, inW, cIv});
                  Value filterValue = rewriter.create<memref::LoadOp>(
                      loc, filter, ValueRange{khIv, kwIv, cIv, ocIv});
                  Value product =
                      rewriter.create<arith::MulFOp>(loc, inValue, filterValue);
                  Value sum =
                      rewriter.create<arith::AddFOp>(loc, cAcc, product);
                  rewriter.create<scf::YieldOp>(loc, sum);
                }
                rewriter.setInsertionPointAfter(cLoop);
                rewriter.create<scf::YieldOp>(loc, cLoop.getResult(0));
              }
              rewriter.setInsertionPointAfter(kwLoop);
              rewriter.create<scf::YieldOp>(loc, kwLoop.getResult(0));
            }
            rewriter.setInsertionPointAfter(khLoop);
            rewriter.create<memref::StoreOp>(loc, khLoop.getResult(0), output,
                                             ValueRange{nIv, ohIv, owIv, ocIv});
          }
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankWidthBytes, bankDepth;
};

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  patterns.add<TileConv2dLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}

namespace {

class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)
  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert Tile dialect to Buckyball dialect";
  }
  LowerTileToBuckyballPass() = default;
  LowerTileToBuckyballPass(const LowerTileToBuckyballPass &) {}

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(kDefaultBankWidthBytes)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Bank depth (rows per bank)."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(16)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::tile::TileDialect,
                    ::buddy::buckyball::BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<::buddy::buckyball::BuckyballDialect,
                           memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
    target.addIllegalDialect<::buddy::tile::TileDialect>();

    RewritePatternSet patterns(context);
    populateLowerTileToBuckyballConversionPatterns(patterns, bankWidthBytes,
                                                   bankDepth, bankNum);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
