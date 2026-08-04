//===- LowerTileToBuckyball.cpp - Pebble tile->buckyball pass -------------===//
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

class TileTransposeLowering : public OpRewritePattern<tile::TileTransposeOp> {
public:
  TileTransposeLowering(MLIRContext *context, int64_t bankWidthBytes)
      : OpRewritePattern(context), bankWidthBytes(bankWidthBytes) {}

  LogicalResult matchAndRewrite(tile::TileTransposeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getAMemArray();
    Value output = op.getBMemArray();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return op.emitError("requires static input and output memrefs");

    auto inShape = inputType.getShape();
    auto outShape = outputType.getShape();
    size_t rows = inShape[0];
    size_t cols = inShape[1];
    if (outShape[0] != static_cast<int64_t>(cols) ||
        outShape[1] != static_cast<int64_t>(rows))
      return op.emitError("output shape must transpose the input shape");

    size_t elemsPerRow =
        elemsPerBankRow(inputType.getElementType(), bankWidthBytes);
    if (elemsPerRow == 0 || cols % elemsPerRow != 0)
      return op.emitError("input width must be a whole number of bank rows");
    int64_t elemBits = inputType.getElementType().getIntOrFloatBitWidth();
    if (elemBits != 8 && elemBits != 32)
      return op.emitError("only i8 and i32 transpose elements are supported");

    constexpr size_t kMaxCols = 64;
    size_t colTile = std::min(cols, kMaxCols);
    colTile = (colTile / elemsPerRow) * elemsPerRow;
    if (colTile == 0)
      return op.emitError("tile width is smaller than one bank row");

    for (size_t r0 = 0; r0 < ceilDiv(rows, kMatmulTile); ++r0) {
      for (size_t c0 = 0; c0 < ceilDiv(cols, colTile); ++c0) {
        size_t rStart = r0 * kMatmulTile;
        size_t cStart = c0 * colTile;
        size_t rLen = std::min(kMatmulTile, rows - rStart);
        size_t cLen = std::min(colTile, cols - cStart);
        size_t paddedRows = std::max(rLen, kMatmulTile);
        int64_t groups = cLen / elemsPerRow;
        if (groups * 2 > 16)
          return op.emitError("transpose tile exceeds Pebble bank capacity");

        Value inTile = rewriter.create<memref::SubViewOp>(
            loc, input,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rStart),
                                      rewriter.getIndexAttr(cStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(rLen),
                                      rewriter.getIndexAttr(cLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});
        Value outTile = rewriter.create<memref::SubViewOp>(
            loc, output,
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cStart),
                                      rewriter.getIndexAttr(rStart)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(cLen),
                                      rewriter.getIndexAttr(rLen)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)});

        Value src = rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
        src.getDefiningOp()->setAttr("col", rewriter.getI64IntegerAttr(groups));
        Value dst = rewriter.create<BankAllocOp>(loc, rewriter.getI64Type());
        dst.getDefiningOp()->setAttr("col", rewriter.getI64IntegerAttr(groups));
        Value loaded = mvinBank(rewriter, loc, inTile, src, paddedRows);
        Value transposed = rewriter.create<BankTransposeOp>(
            loc, dst.getType(), loaded, dst,
            createI64Const(rewriter, loc, paddedRows),
            createI64Const(rewriter, loc, elemBits));
        mvoutBank(rewriter, loc, outTile, transposed, paddedRows);
        releaseBank(rewriter, loc, loaded);
        releaseBank(rewriter, loc, transposed);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t bankWidthBytes;
};

class TileConv2dLowering : public OpRewritePattern<tile::TileConv2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileConv2dOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto filterTy = dyn_cast<MemRefType>(op.getFilter().getType());
    auto outputTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputTy || !filterTy || !outputTy || !inputTy.hasStaticShape() ||
        !filterTy.hasStaticShape() || !outputTy.hasStaticShape() ||
        !inputTy.getElementType().isF32() ||
        !filterTy.getElementType().isF32() ||
        !outputTy.getElementType().isF32())
      return op.emitError("requires static f32 memrefs");
    if (inputTy.getShape() != ArrayRef<int64_t>{1, 6, 6, 1} ||
        filterTy.getShape() != ArrayRef<int64_t>{3, 3, 1, 1} ||
        outputTy.getShape() != ArrayRef<int64_t>{1, 4, 4, 1})
      return op.emitError("currently supports NHWC 1x6x6x1 and HWCF 3x3x1x1");

    auto inputPackTy = MemRefType::get({3, 16}, rewriter.getF32Type());
    auto filterPackTy = MemRefType::get({16, 16}, rewriter.getF32Type());
    auto outputPackTy = MemRefType::get({16, 16}, rewriter.getF32Type());
    Value inputPack = rewriter.create<memref::AllocOp>(loc, inputPackTy);
    Value filterPack = rewriter.create<memref::AllocOp>(loc, filterPackTy);
    Value outputPack = rewriter.create<memref::AllocOp>(loc, outputPackTy);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value six = rewriter.create<arith::ConstantIndexOp>(loc, 6);
    Value three = rewriter.create<arith::ConstantIndexOp>(loc, 3);
    Value four = rewriter.create<arith::ConstantIndexOp>(loc, 4);
    Value sixteen = rewriter.create<arith::ConstantIndexOp>(loc, 16);
    Value fzero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0));
    rewriter.create<linalg::FillOp>(loc, fzero, inputPack);
    rewriter.create<linalg::FillOp>(loc, fzero, filterPack);

    auto inRow = rewriter.create<scf::ForOp>(loc, zero, six, one);
    rewriter.setInsertionPointToStart(inRow.getBody());
    auto inCol = rewriter.create<scf::ForOp>(loc, zero, six, one);
    rewriter.setInsertionPointToStart(inCol.getBody());
    Value flat = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, inRow.getInductionVar(), six),
        inCol.getInductionVar());
    Value packRow = rewriter.create<arith::DivUIOp>(loc, flat, sixteen);
    Value packCol = rewriter.create<arith::RemUIOp>(loc, flat, sixteen);
    Value inputValue = rewriter.create<memref::LoadOp>(
        loc, op.getInput(),
        ValueRange{zero, inRow.getInductionVar(), inCol.getInductionVar(),
                   zero});
    rewriter.create<memref::StoreOp>(loc, inputValue, inputPack,
                                     ValueRange{packRow, packCol});
    rewriter.setInsertionPointAfter(inRow);

    auto filterRow = rewriter.create<scf::ForOp>(loc, zero, three, one);
    rewriter.setInsertionPointToStart(filterRow.getBody());
    auto filterCol = rewriter.create<scf::ForOp>(loc, zero, three, one);
    rewriter.setInsertionPointToStart(filterCol.getBody());
    Value filterIndex = rewriter.create<arith::AddIOp>(
        loc,
        rewriter.create<arith::MulIOp>(loc, filterRow.getInductionVar(), three),
        filterCol.getInductionVar());
    Value filterValue = rewriter.create<memref::LoadOp>(
        loc, op.getFilter(),
        ValueRange{filterRow.getInductionVar(), filterCol.getInductionVar(),
                   zero, zero});
    rewriter.create<memref::StoreOp>(loc, filterValue, filterPack,
                                     ValueRange{filterIndex, zero});
    rewriter.setInsertionPointAfter(filterRow);

    Value inputF = allocBank(rewriter, loc, 1, 4);
    Value inputI = allocBank(rewriter, loc, 1, 1);
    Value scale = createI64Const(rewriter, loc, 1065353216);
    Value inputLoaded = mvinBank(rewriter, loc, inputPack, inputF, 3);
    Value inputQuant = rewriter.create<BankFp2IntOp>(
        loc, inputI.getType(), inputLoaded, inputI,
        createI64Const(rewriter, loc, 3), scale);
    releaseBank(rewriter, loc, inputLoaded);

    Value patches = allocBank(rewriter, loc, 1, 1);
    Value patchBank = rewriter.create<BankIm2colOp>(
        loc, patches.getType(), inputQuant, patches,
        createI64Const(rewriter, loc, 6), createI64Const(rewriter, loc, 3),
        createI64Const(rewriter, loc, 1), createI64Const(rewriter, loc, 0));
    releaseBank(rewriter, loc, inputQuant);

    Value filterF = allocBank(rewriter, loc, 1, 4);
    Value filterI = allocBank(rewriter, loc, 1, 1);
    Value filterLoaded = mvinBank(rewriter, loc, filterPack, filterF, 16);
    Value filterQuant = rewriter.create<BankFp2IntOp>(
        loc, filterI.getType(), filterLoaded, filterI,
        createI64Const(rewriter, loc, 16), scale);
    releaseBank(rewriter, loc, filterLoaded);

    Value acc = allocBank(rewriter, loc, 1, 4);
    Value computed = rewriter.create<BankMatrixOp>(
        loc, acc.getType(), patchBank, filterQuant, acc,
        createI64Const(rewriter, loc, 0x09001010));
    releaseBank(rewriter, loc, patchBank);
    releaseBank(rewriter, loc, filterQuant);

    Value outputF = allocBank(rewriter, loc, 1, 4);
    Value dequant =
        rewriter.create<BankInt2FpOp>(loc, outputF.getType(), computed, outputF,
                                      createI64Const(rewriter, loc, 16), scale);
    releaseBank(rewriter, loc, computed);
    Value outputStored = mvoutBank(rewriter, loc, outputPack, dequant, 16);

    auto outRow = rewriter.create<scf::ForOp>(loc, zero, four, one);
    rewriter.setInsertionPointToStart(outRow.getBody());
    auto outCol = rewriter.create<scf::ForOp>(loc, zero, four, one);
    rewriter.setInsertionPointToStart(outCol.getBody());
    Value outIndex = rewriter.create<arith::AddIOp>(
        loc,
        rewriter.create<arith::MulIOp>(loc, outRow.getInductionVar(), four),
        outCol.getInductionVar());
    Value outputValue = rewriter.create<memref::LoadOp>(
        loc, outputPack, ValueRange{outIndex, zero});
    rewriter.create<memref::StoreOp>(loc, outputValue, op.getOutput(),
                                     ValueRange{zero, outRow.getInductionVar(),
                                                outCol.getInductionVar(),
                                                zero});
    rewriter.setInsertionPointAfter(outRow);
    releaseBank(rewriter, loc, outputStored);
    rewriter.create<memref::DeallocOp>(loc, inputPack);
    rewriter.create<memref::DeallocOp>(loc, filterPack);
    rewriter.create<memref::DeallocOp>(loc, outputPack);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::populateLowerTileToBuckyballConversionPatterns(
    RewritePatternSet &patterns, int64_t bankWidthBytes, int64_t bankDepth,
    int64_t bankNum) {
  populateMatrixTileMatMulPatterns(patterns, bankWidthBytes, bankDepth,
                                   bankNum);
  patterns.add<TileTransposeLowering>(patterns.getContext(), bankWidthBytes);
  patterns.add<TileConv2dLowering>(patterns.getContext());
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
    target.addIllegalOp<::buddy::tile::TileMatMulOp>();
    target.addIllegalOp<::buddy::tile::TileConv2dOp>();
    target.addIllegalOp<::buddy::tile::TileTransposeOp>();

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
