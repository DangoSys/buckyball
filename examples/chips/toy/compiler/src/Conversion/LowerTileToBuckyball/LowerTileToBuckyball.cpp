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
#include "Buckyball/BuckyballOps.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"
#include "Tile/Transform.h"

#include "Utils/BankUtils.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::tile;
namespace tile = ::buddy::tile;
using mlir::buddy::ceilDiv;
using mlir::buddy::kDefaultBankWidthBytes;
using mlir::buddy::kMatmulTile;
using mlir::buddy::populateMatrixTileMatMulPatterns;

namespace {

static size_t elemsPerBankRow(Type elemType, size_t bankWidthBytes) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return 0;
  return bankWidthBytes / (bitWidth / 8);
}

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(v));
}

static Value packF32BitsAsI64(OpBuilder &b, Location loc, Value f32Val) {
  Value i32Bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), f32Val);
  return b.create<arith::ExtUIOp>(loc, b.getI64Type(), i32Bits);
}

static Value buildTileAbsMax(PatternRewriter &rewriter, Location loc, Value mem,
                             uint64_t rows, uint64_t cols) {
  auto maxTy = MemRefType::get({1}, rewriter.getF32Type());
  Value maxBuf = rewriter.create<memref::AllocOp>(loc, maxTy);

  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value rowsIdx = rewriter.create<arith::ConstantIndexOp>(loc, rows);
  Value colsIdx = rewriter.create<arith::ConstantIndexOp>(loc, cols);
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);

  rewriter.create<memref::StoreOp>(loc, zeroF32, maxBuf, ValueRange{zeroIdx});

  auto rowLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, rowsIdx, oneIdx);
  rewriter.setInsertionPointToStart(rowLoop.getBody());
  auto colLoop = rewriter.create<scf::ForOp>(loc, zeroIdx, colsIdx, oneIdx);
  rewriter.setInsertionPointToStart(colLoop.getBody());

  Value elem = rewriter.create<memref::LoadOp>(
      loc, mem,
      ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});
  if (elem.getType() != rewriter.getF32Type())
    elem = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), elem);
  Value neg = rewriter.create<arith::NegFOp>(loc, elem);
  Value abs = rewriter.create<arith::MaximumFOp>(loc, elem, neg);
  Value cur = rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  Value upd = rewriter.create<arith::MaximumFOp>(loc, cur, abs);
  rewriter.create<memref::StoreOp>(loc, upd, maxBuf, ValueRange{zeroIdx});

  rewriter.setInsertionPointAfter(rowLoop);
  Value result =
      rewriter.create<memref::LoadOp>(loc, maxBuf, ValueRange{zeroIdx});
  rewriter.create<memref::DeallocOp>(loc, maxBuf);
  return result;
}

static Value buildQuantScale(PatternRewriter &rewriter, Location loc,
                             Value maxAbs) {
  Value zeroF32 = cstF32(rewriter, loc, 0.0f);
  Value oneF32 = cstF32(rewriter, loc, 1.0f);
  Value qmaxF32 = cstF32(rewriter, loc, 127.0f);
  Value hasData = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                 maxAbs, zeroF32);
  Value scaled = rewriter.create<arith::DivFOp>(loc, qmaxF32, maxAbs);
  return rewriter.create<arith::SelectOp>(loc, hasData, scaled, oneF32);
}

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  explicit TileTransposeLowering(MLIRContext *context, int64_t bankWidthBytes,
                                 int64_t /*bankDepth*/, int64_t /*bankNum*/)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp tileTransposeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = tileTransposeOp.getLoc();

    Value inputMemArray = tileTransposeOp.getAMemArray();
    Value outputMemArray = tileTransposeOp.getBMemArray();

    auto inputType = cast<MemRefType>(inputMemArray.getType());
    auto outputType = cast<MemRefType>(outputMemArray.getType());
    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();

    size_t Rows = inShape[inShape.size() - 2];
    size_t Cols = inShape[inShape.size() - 1];

    if (outShape[outShape.size() - 2] != (int64_t)Cols ||
        outShape[outShape.size() - 1] != (int64_t)Rows)
      return tileTransposeOp.emitError(
          "Output shape must be transposed of input shape");

    size_t elemsPerRow =
        elemsPerBankRow(inputType.getElementType(), bankWidthBytes);
    if (elemsPerRow == 0)
      return tileTransposeOp.emitError("unsupported transpose element type");

    constexpr size_t kTransposeRows = kMatmulTile;
    constexpr size_t kMaxTransposeCols = 64;

    size_t colTileSize = std::min(Cols, kMaxTransposeCols);
    colTileSize = (colTileSize / elemsPerRow) * elemsPerRow;
    if (colTileSize == 0)
      colTileSize = elemsPerRow;

    size_t rowTileNum = ceilDiv(Rows, kTransposeRows);
    size_t colTileNum = ceilDiv(Cols, colTileSize);

    for (size_t r0 = 0; r0 < rowTileNum; r0++) {
      for (size_t c0 = 0; c0 < colTileNum; c0++) {
        size_t rStart = r0 * kTransposeRows;
        size_t cStart = c0 * colTileSize;
        size_t rLen = std::min(kTransposeRows, Rows - rStart);
        size_t cLen = std::min(colTileSize, Cols - cStart);
        size_t rLenPadded = (rLen < kTransposeRows) ? kTransposeRows : rLen;

        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, inputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rStart),
                                      rewriter.getIndexAttr(cStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rLen),
                                      rewriter.getIndexAttr(cLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, outputMemArray,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cStart),
                                      rewriter.getIndexAttr(rStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cLen),
                                      rewriter.getIndexAttr(rLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        Value srcBank =
            rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
        Value dstBank =
            rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());

        int64_t depth = rLenPadded * cLen / elemsPerRow;
        Value srcBankAfterMvin =
            mvinBank(rewriter, loc, inTile, srcBank, depth);

        Value iterVal = createI64Const(rewriter, loc, cLen);
        Value modeVal = createI64Const(rewriter, loc, 0);
        Value dstBankAfterTranspose = rewriter.create<BankTransposeOp>(
            loc, dstBank.getType(), srcBankAfterMvin, dstBank, iterVal,
            modeVal);

        int64_t outDepth = cLen * rLen / elemsPerRow;
        mvoutBank(rewriter, loc, outTile, dstBankAfterTranspose, outDepth);

        releaseBank(rewriter, loc, srcBankAfterMvin);
        releaseBank(rewriter, loc, dstBankAfterTranspose);
      }
    }

    rewriter.eraseOp(tileTransposeOp);
    return success();
  }

private:
  int64_t bankWidthBytes;
};

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

void populateToyLocalTilePatterns(RewritePatternSet &patterns,
                                  int64_t bankWidthBytes, int64_t bankDepth,
                                  int64_t bankNum) {
  patterns.add<TileTransposeLowering>(patterns.getContext(), bankWidthBytes,
                                      bankDepth, bankNum);
  patterns.add<TileConv2dLowering>(patterns.getContext(), bankWidthBytes,
                                   bankDepth, bankNum);
}

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  populateMatrixTileMatMulPatterns(patterns, bankWidthBytes, bankDepth,
                                   bankNum);
  populateToyLocalTilePatterns(patterns, bankWidthBytes, bankDepth, bankNum);
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
                            llvm::cl::init(4096)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};

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
