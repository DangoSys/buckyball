//===- LegalizeForLLVMExportBase.cpp - Buckyball base LLVM lowering -------===//
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

#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace buddy {
namespace buckyball {
namespace legalize {

uint64_t fieldBits(uint64_t val, int startBit, int endBit) {
  uint64_t width = endBit - startBit + 1;
  uint64_t mask = (1ULL << width) - 1;
  return (val & mask) << startBit;
}

Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(),
                                     b.getI64IntegerAttr(v));
}

static int64_t elemByteSize(Type el) {
  if (auto it = dyn_cast<IntegerType>(el))
    return it.getWidth() / 8;
  if (auto ft = dyn_cast<FloatType>(el))
    return ft.getWidth() / 8;
  return -1;
}

struct MemrefAddress {
  Value address;
  Value hostPtr;
};

static MemrefAddress extractMemrefAddress(OpBuilder &b, Location loc,
                                          Value memref) {
  auto ty = cast<MemRefType>(memref.getType());
  int64_t eb = elemByteSize(ty.getElementType());
  if (eb <= 0)
    llvm_unreachable(
        "bb memref intrinsic: unsupported element type for ptr offset");
  auto meta = b.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value base = meta.getBaseBuffer();
  Value off = meta.getOffset();
  Value baseIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, b.getIndexType(), base);
  Value baseI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), baseIdx);
  Value offI64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(), off);
  Value offBytes = offI64;
  if (eb != 1)
    offBytes = b.create<arith::MulIOp>(loc, offI64, cstI64(b, loc, eb));
  Value address = b.create<arith::AddIOp>(loc, baseI64, offBytes);
  Type ptrType = LLVM::LLVMPointerType::get(b.getContext());
  Value hostPtr = LLVM::IntToPtrOp::create(b, loc, ptrType, address);
  return {address, hostPtr};
}

Value extractPtr(OpBuilder &b, Location loc, Value memref) {
  return extractMemrefAddress(b, loc, memref).address;
}

Value packRs1BanksIter(OpBuilder &b, Location loc, Value rBank0, Value rBank1,
                       Value wBank, Value iter) {
  Value rBank0Field =
      b.create<arith::AndIOp>(loc, rBank0, cstI64(b, loc, 0x3FF));
  Value rBank1Field = b.create<arith::ShLIOp>(
      loc, b.create<arith::AndIOp>(loc, rBank1, cstI64(b, loc, 0x3FF)),
      cstI64(b, loc, 10));
  Value wBankField = b.create<arith::ShLIOp>(
      loc, b.create<arith::AndIOp>(loc, wBank, cstI64(b, loc, 0x3FF)),
      cstI64(b, loc, 20));
  Value iterField = b.create<arith::ShLIOp>(
      loc, b.create<arith::AndIOp>(loc, iter, cstI64(b, loc, (1ULL << 34) - 1)),
      cstI64(b, loc, 30));
  Value rs1Part01 = b.create<arith::OrIOp>(loc, rBank0Field, rBank1Field);
  Value rs1Part012 = b.create<arith::OrIOp>(loc, rs1Part01, wBankField);
  return b.create<arith::OrIOp>(loc, rs1Part012, iterField);
}

Value packRs1BankIter(OpBuilder &b, Location loc, Value bankId, Value depth) {
  Value z = cstI64(b, loc, 0);
  return packRs1BanksIter(b, loc, bankId, z, z, depth);
}

Value packRs2MemStride(OpBuilder &b, Location loc, Value memAddr,
                       Value stride) {
  Value mem =
      b.create<arith::AndIOp>(loc, memAddr, cstI64(b, loc, (1ULL << 39) - 1));
  Value s =
      b.create<arith::AndIOp>(loc, stride, cstI64(b, loc, (1ULL << 19) - 1));
  Value sHi = b.create<arith::ShLIOp>(loc, s, cstI64(b, loc, 39));
  return b.create<arith::OrIOp>(loc, mem, sHi);
}

void emitMset(OpBuilder &b, Location loc, uint64_t bankId, uint64_t row,
              uint64_t col, uint64_t alloc) {
  uint64_t rs1 = fieldBits(bankId, 0, 9);
  uint64_t rs2 =
      fieldBits(row, 0, 4) | fieldBits(col, 5, 9) | fieldBits(alloc, 10, 10);
  b.create<MsetIntrOp>(loc, cstI64(b, loc, rs1), cstI64(b, loc, rs2));
}

static constexpr char kBbDmaTouchMvoutFn[] = "bb_dma_touch_mvout";
static constexpr char kBbDmaBankSetColsFn[] = "bb_dma_bank_set_cols";

static FlatSymbolRefAttr getOrInsertExtFunc(OpBuilder &b, ModuleOp module,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return FlatSymbolRefAttr::get(b.getContext(), name);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(b, module.getLoc(), name, type,
                           LLVM::Linkage::External, false, LLVM::CConv::C);
  return FlatSymbolRefAttr::get(b.getContext(), name);
}

static ModuleOp parentModule(OpBuilder &b) {
  Operation *op = b.getInsertionBlock()->getParentOp();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    llvm_unreachable("bb_dma: missing ModuleOp parent");
  return module;
}

static void emitBbDmaBankSetCols(OpBuilder &b, Location loc, Value bankId,
                                 Value cols) {
  ModuleOp module = parentModule(b);
  auto i32Ty = IntegerType::get(b.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(b.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
  FlatSymbolRefAttr callee =
      getOrInsertExtFunc(b, module, kBbDmaBankSetColsFn, fnTy);
  Value bankI32 = b.create<arith::TruncIOp>(loc, i32Ty, bankId);
  Value colsI32 = b.create<arith::TruncIOp>(loc, i32Ty, cols);
  LLVM::CallOp::create(b, loc, TypeRange{}, callee,
                       ValueRange{bankI32, colsI32});
}

static void emitBbDmaTouchMvout(OpBuilder &b, Location loc, Value hostPtr,
                                Value depth, Value stride, Value bankId) {
  ModuleOp module = parentModule(b);
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext());
  auto i64Ty = IntegerType::get(b.getContext(), 64);
  auto i32Ty = IntegerType::get(b.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(b.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {ptrTy, i64Ty, i64Ty, i32Ty});
  FlatSymbolRefAttr callee =
      getOrInsertExtFunc(b, module, kBbDmaTouchMvoutFn, fnTy);
  Value bankI32 = b.create<arith::TruncIOp>(loc, i32Ty, bankId);
  LLVM::CallOp::create(b, loc, TypeRange{}, callee,
                       ValueRange{hostPtr, depth, stride, bankI32});
}

namespace {

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct BuckyballFenceLowering : public ConvertOpToLLVMPattern<FenceOp> {
  using ConvertOpToLLVMPattern<FenceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(FenceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = cstI64(rewriter, loc, 0);
    rewriter.replaceOpWithNewOp<FenceIntrOp>(op, zero, zero);
    return success();
  }
};

struct BuckyballMsetLowering : public ConvertOpToLLVMPattern<MsetOp> {
  using ConvertOpToLLVMPattern<MsetOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bankId = adaptor.getBankId();
    Value rs1 = rewriter.create<arith::AndIOp>(loc, bankId,
                                               cstI64(rewriter, loc, 0x3FF));
    uint64_t allocBit = op.getAlloc() ? 1u : 0u;
    uint64_t rowVal = op.getAlloc() ? static_cast<uint64_t>(op.getRow()) : 0u;
    uint64_t colVal = op.getAlloc() ? static_cast<uint64_t>(op.getCol()) : 0u;
    uint64_t rs2Val = fieldBits(rowVal, 0, 4) | fieldBits(colVal, 5, 9) |
                      fieldBits(allocBit, 10, 10);
    Value colsForTouch = cstI64(rewriter, loc, op.getAlloc() ? colVal : 1u);
    emitBbDmaBankSetCols(rewriter, loc, bankId, colsForTouch);
    rewriter.replaceOpWithNewOp<MsetIntrOp>(op, rs1,
                                            cstI64(rewriter, loc, rs2Val));
    return success();
  }
};

struct BuckyballMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  BuckyballMvinLowering(LLVMTypeConverter &converter, bool rushB)
      : ConvertOpToLLVMPattern<MvinOp>(converter), rushB(rushB) {}

  LogicalResult
  matchAndRewrite(MvinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getInput());
    Value rs1 =
        packRs1BankIter(rewriter, loc, adaptor.getAddr(), adaptor.getDepth());
    Value rs2 =
        packRs2MemStride(rewriter, loc, memref.address, adaptor.getStride());
    if (rushB) {
      rewriter.replaceOpWithNewOp<RushBMvinOp>(op, rs1, rs2, memref.hostPtr);
      return success();
    }
    rewriter.replaceOpWithNewOp<MvinIntrOp>(op, rs1, rs2);
    return success();
  }

private:
  bool rushB;
};

struct BuckyballMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  BuckyballMvoutLowering(LLVMTypeConverter &converter, bool rushB)
      : ConvertOpToLLVMPattern<MvoutOp>(converter), rushB(rushB) {}

  LogicalResult
  matchAndRewrite(MvoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getOutput());
    emitBbDmaTouchMvout(rewriter, loc, memref.hostPtr, adaptor.getDepth(),
                        adaptor.getStride(), adaptor.getAddr());
    Value rs1 =
        packRs1BankIter(rewriter, loc, adaptor.getAddr(), adaptor.getDepth());
    Value rs2 =
        packRs2MemStride(rewriter, loc, memref.address, adaptor.getStride());
    if (rushB) {
      rewriter.replaceOpWithNewOp<RushBMvoutOp>(op, rs1, rs2, memref.hostPtr);
      return success();
    }
    rewriter.replaceOpWithNewOp<MvoutIntrOp>(op, rs1, rs2);
    return success();
  }

private:
  bool rushB;
};

struct BuckyballInstLowering : public ConvertOpToLLVMPattern<InstOp> {
  using ConvertOpToLLVMPattern<InstOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(InstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, adaptor.getRs1(), adaptor.getRs2(),
        rewriter.getI32IntegerAttr(op.getFunct7()));
    return success();
  }
};

} // namespace

void populateBaseLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool includeFuncOperandForwarding, bool rushB) {
  if (includeFuncOperandForwarding) {
    patterns.add<ForwardOperands<func::CallOp>,
                 ForwardOperands<func::CallIndirectOp>,
                 ForwardOperands<func::ReturnOp>>(converter,
                                                  &converter.getContext());
  }
  patterns.add<BuckyballFenceLowering>(converter);
  patterns.add<BuckyballMsetLowering>(converter);
  patterns.add<BuckyballMvinLowering>(converter, rushB);
  patterns.add<BuckyballMvoutLowering>(converter, rushB);
  patterns.add<BuckyballInstLowering>(converter);
}

void configureBaseLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<CustomIntrOp, FenceIntrOp, MsetIntrOp, MvinIntrOp,
                    MvoutIntrOp, RushBMvinOp, RushBMvoutOp>();
  target.addIllegalOp<FenceOp, InstOp, MsetOp, MvinOp, MvoutOp, BankAllocOp,
                      BankReleaseOp, BankMvinOp, BankMvoutOp>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
}

} // namespace legalize
} // namespace buckyball
} // namespace buddy
