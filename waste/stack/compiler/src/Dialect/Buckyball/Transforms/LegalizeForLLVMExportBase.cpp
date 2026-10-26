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

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace buddy {
namespace buckyball {
namespace legalize {

uint64_t fieldBits(uint64_t val, int startBit, int endBit) {
  uint64_t width = endBit - startBit + 1;
  uint64_t mask = (1ULL << width) - 1;
  return (val & mask) << startBit;
}

int64_t addrBitsForDepth(int64_t bankDepth) {
  if (bankDepth <= 1)
    return -1;
  int64_t bits = 0;
  int64_t x = bankDepth - 1;
  while (x) {
    x >>= 1;
    ++bits;
  }
  return bits;
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

Value packRs1WriteBankIter(OpBuilder &b, Location loc, Value bankId,
                           Value depth) {
  Value z = cstI64(b, loc, 0);
  return packRs1BanksIter(b, loc, z, z, bankId, depth);
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

static constexpr char kDmaTouchMvoutFn[] = "dma_touch_mvout";

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

static void emitBbDmaBankColumns(OpBuilder &b, Location loc, StringRef name,
                                 Value bankId, Value cols) {
  ModuleOp module = parentModule(b);
  auto i32Ty = IntegerType::get(b.getContext(), 32);
  auto voidTy = LLVM::LLVMVoidType::get(b.getContext());
  auto fnTy = LLVM::LLVMFunctionType::get(voidTy, {i32Ty, i32Ty});
  FlatSymbolRefAttr callee = getOrInsertExtFunc(b, module, name, fnTy);
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
      getOrInsertExtFunc(b, module, kDmaTouchMvoutFn, fnTy);
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
    Value colsForTouch = cstI64(rewriter, loc, op.getAlloc() ? colVal : 0u);
    emitBbDmaBankColumns(rewriter, loc,
                         op.getAlloc() ? "dma_bank_allocate"
                                       : "dma_bank_set_cols",
                         bankId, colsForTouch);
    rewriter.replaceOpWithNewOp<MsetIntrOp>(op, rs1,
                                            cstI64(rewriter, loc, rs2Val));
    return success();
  }
};

struct BuckyballMsetTransferLowering
    : public ConvertOpToLLVMPattern<MsetTransferOp> {
  using ConvertOpToLLVMPattern<MsetTransferOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MsetTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource(), target = adaptor.getTarget();
    auto i64 = rewriter.getI64Type();
    auto fnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), {i64, i64});
    auto callee = getOrInsertExtFunc(rewriter, parentModule(rewriter),
                                     "dma_bank_transfer", fnType);
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, callee,
                         ValueRange{source, target});
    Value targetBits =
        rewriter.create<arith::ShLIOp>(loc, target, cstI64(rewriter, loc, 20));
    Value rs1 = rewriter.create<arith::OrIOp>(loc, source, targetBits);
    rewriter.replaceOpWithNewOp<MsetIntrOp>(op, rs1,
                                            cstI64(rewriter, loc, 4096));
    return success();
  }
};

struct BuckyballMvinLowering : public ConvertOpToLLVMPattern<MvinOp> {
  using ConvertOpToLLVMPattern<MvinOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MvinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getInput());
    Value rs1 = packRs1WriteBankIter(rewriter, loc, adaptor.getAddr(),
                                     adaptor.getDepth());
    Value rs2 =
        packRs2MemStride(rewriter, loc, memref.address, adaptor.getStride());
    if (auto group = op.getGroup())
      rs2 = rewriter.create<arith::OrIOp>(
          loc, rs2,
          cstI64(rewriter, loc, (1ULL << 63) | (uint64_t(*group) << 58)));

    rewriter.replaceOpWithNewOp<MvinIntrOp>(op, rs1, rs2);
    return success();
  }
};

struct BuckyballMvin2dLowering : public ConvertOpToLLVMPattern<Mvin2dOp> {
  using ConvertOpToLLVMPattern<Mvin2dOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mvin2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getInput());
    Value lowBits = rewriter.create<arith::AndIOp>(loc, memref.address,
                                                   cstI64(rewriter, loc, 7));
    Value aligned = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lowBits, cstI64(rewriter, loc, 0));
    Value inRange = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, memref.address,
        cstI64(rewriter, loc, 1ULL << 39));
    Value valid8 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  adaptor.getValidBytes(),
                                                  cstI64(rewriter, loc, 8));
    Value valid16 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, adaptor.getValidBytes(),
        cstI64(rewriter, loc, 16));
    Value validLength = rewriter.create<arith::OrIOp>(loc, valid8, valid16);
    Value validAddress = rewriter.create<arith::AndIOp>(loc, aligned, inRange);
    Value validInput =
        rewriter.create<arith::AndIOp>(loc, validAddress, validLength);
    Value validInput64 =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), validInput);
    // This lowering also runs inside single-block SCF regions.
    auto tail = LLVM::TailCallKindAttr::get(
        rewriter.getContext(), LLVM::tailcallkind::TailCallKind::None);
    LLVM::InlineAsmOp::create(
        rewriter, loc, Type(), ValueRange{validInput64},
        rewriter.getStringAttr("bnez $0, 1f\n\tunimp\n1:"),
        rewriter.getStringAttr("r,~{memory}"), rewriter.getUnitAttr(),
        UnitAttr(), tail, UnitAttr(), nullptr, nullptr);
    Value rs1 = packRs1WriteBankIter(rewriter, loc, adaptor.getAddr(),
                                     adaptor.getHeight());
    Value address = rewriter.create<arith::ShRUIOp>(loc, memref.address,
                                                    cstI64(rewriter, loc, 3));
    Value pixel = rewriter.create<arith::AndIOp>(loc, adaptor.getPixelBytes(),
                                                 cstI64(rewriter, loc, 0x3ff));
    Value pixelUnits =
        rewriter.create<arith::DivUIOp>(loc, pixel, cstI64(rewriter, loc, 8));
    Value width = rewriter.create<arith::SubIOp>(loc, adaptor.getWidth(),
                                                 cstI64(rewriter, loc, 1));
    Value valid =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), valid8);
    Value rs2 = rewriter.create<arith::OrIOp>(
        loc, address,
        rewriter.create<arith::ShLIOp>(loc, pixelUnits,
                                       cstI64(rewriter, loc, 36)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getSourceWidth(),
                                       cstI64(rewriter, loc, 43)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getDstBase(),
                                       cstI64(rewriter, loc, 53)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, width, cstI64(rewriter, loc, 59)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, valid, cstI64(rewriter, loc, 62)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, rs2,
                                              rewriter.getI32IntegerAttr(34));
    return success();
  }
};

struct BuckyballMvinMmioLowering : public ConvertOpToLLVMPattern<MvinMmioOp> {
  using ConvertOpToLLVMPattern<MvinMmioOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvinMmioOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getInput());
    Value rs1 = rewriter.create<arith::ShLIOp>(loc, adaptor.getRows(),
                                               cstI64(rewriter, loc, 30));
    Value address = rewriter.create<arith::AndIOp>(
        loc, memref.address, cstI64(rewriter, loc, (1ULL << 39) - 1));
    Value mmio = rewriter.create<arith::AndIOp>(
        loc, adaptor.getMmioAddr(), cstI64(rewriter, loc, (1ULL << 17) - 1));
    Value columns = rewriter.create<arith::AndIOp>(loc, adaptor.getColumns(),
                                                   cstI64(rewriter, loc, 0xff));
    Value rs2 = rewriter.create<arith::OrIOp>(
        loc, address,
        rewriter.create<arith::ShLIOp>(loc, mmio, cstI64(rewriter, loc, 39)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, columns,
                                       cstI64(rewriter, loc, 56)));
    rewriter.replaceOpWithNewOp<MvinMmioIntrOp>(op, rs1, rs2);
    return success();
  }
};

struct BuckyballMvoutLowering : public ConvertOpToLLVMPattern<MvoutOp> {
  using ConvertOpToLLVMPattern<MvoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MvoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MemrefAddress memref = extractMemrefAddress(rewriter, loc, op.getOutput());
    if (op.getGroup()) {
      auto fnTy = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(rewriter.getContext()),
          {LLVM::LLVMPointerType::get(rewriter.getContext()),
           rewriter.getI64Type(), rewriter.getI64Type()});
      auto callee = getOrInsertExtFunc(rewriter, parentModule(rewriter),
                                       "dma_touch_mvout_group", fnTy);
      LLVM::CallOp::create(
          rewriter, loc, TypeRange{}, callee,
          ValueRange{memref.hostPtr, adaptor.getDepth(), adaptor.getStride()});
    } else {
      emitBbDmaTouchMvout(rewriter, loc, memref.hostPtr, adaptor.getDepth(),
                          adaptor.getStride(), adaptor.getAddr());
    }
    Value rs1 =
        packRs1BankIter(rewriter, loc, adaptor.getAddr(), adaptor.getDepth());
    Value rs2 =
        packRs2MemStride(rewriter, loc, memref.address, adaptor.getStride());
    if (auto group = op.getGroup())
      rs2 = rewriter.create<arith::OrIOp>(
          loc, rs2,
          cstI64(rewriter, loc, (1ULL << 63) | (uint64_t(*group) << 58)));

    rewriter.replaceOpWithNewOp<MvoutIntrOp>(op, rs1, rs2);
    return success();
  }
};

struct BuckyballMvoverLowering : public ConvertOpToLLVMPattern<MvoverOp> {
  using ConvertOpToLLVMPattern<MvoverOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MvoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto bounded = [&](Value value, uint64_t maximum) {
      Value valid =
          rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, value,
                                         cstI64(rewriter, loc, maximum));
      rewriter.create<cf::AssertOp>(loc, valid,
                                    "mvover operand exceeds ISA range");
    };
    bounded(adaptor.getSourceThread(), 255);
    bounded(adaptor.getTargetThread(), 255);
    bounded(adaptor.getSourceBank(), 1023);
    bounded(adaptor.getTargetBank(), 1023);
    bounded(adaptor.getSourceRow(), 65535);
    bounded(adaptor.getTargetRow(), 65535);
    Value count = rewriter.create<arith::SubIOp>(loc, adaptor.getRows(),
                                                 cstI64(rewriter, loc, 1));
    bounded(count, 65535);
    auto shifted = [&](Value value, uint64_t shift) -> Value {
      return rewriter.create<arith::ShLIOp>(loc, value,
                                            cstI64(rewriter, loc, shift));
    };
    Value rs1 = rewriter.create<arith::OrIOp>(
        loc, adaptor.getSourceThread(), shifted(adaptor.getTargetThread(), 8));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1,
                                        shifted(adaptor.getSourceBank(), 16));
    rs1 = rewriter.create<arith::OrIOp>(loc, rs1,
                                        shifted(adaptor.getTargetBank(), 26));
    Value rs2 = rewriter.create<arith::OrIOp>(
        loc, adaptor.getSourceRow(), shifted(adaptor.getTargetRow(), 16));
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2, shifted(count, 32));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, rs2,
                                              rewriter.getI32IntegerAttr(13));
    return success();
  }
};

static bool contiguousRankOne(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(memref.getStridesAndOffset(strides, offset)) &&
         strides[0] == 1;
}

struct ContiguousCopyLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  using ConvertOpToLLVMPattern<memref::CopyOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!contiguousRankOne(op.getSource().getType()) ||
        !contiguousRankOne(op.getTarget().getType()))
      return failure();
    Location loc = op.getLoc();
    auto sourceType = cast<MemRefType>(op.getSource().getType());
    auto targetType = cast<MemRefType>(op.getTarget().getType());
    Value zero =
        LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                 rewriter.getIntegerAttr(getIndexType(), 0));
    Value source = getStridedElementPtr(rewriter, loc, sourceType,
                                        adaptor.getSource(), ValueRange{zero});
    Value target = getStridedElementPtr(rewriter, loc, targetType,
                                        adaptor.getTarget(), ValueRange{zero});
    Value size = MemRefDescriptor(adaptor.getSource()).size(rewriter, loc, 0);
    Value bytes = LLVM::MulOp::create(
        rewriter, loc, size,
        getSizeInBytes(loc, sourceType.getElementType(), rewriter));
    LLVM::MemcpyOp::create(rewriter, loc, target, source, bytes, false);
    rewriter.eraseOp(op);
    return success();
  }
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
    bool includeFuncOperandForwarding) {
  if (includeFuncOperandForwarding) {
    patterns.add<ForwardOperands<func::CallOp>,
                 ForwardOperands<func::CallIndirectOp>,
                 ForwardOperands<func::ReturnOp>>(converter,
                                                  &converter.getContext());
  }
  patterns.add<ContiguousCopyLowering>(converter);
  patterns.add<BuckyballMsetLowering, BuckyballMsetTransferLowering>(converter);
  patterns.add<BuckyballMvinLowering>(converter);
  patterns.add<BuckyballMvin2dLowering>(converter);
  patterns.add<BuckyballMvinMmioLowering>(converter);
  patterns.add<BuckyballMvoutLowering>(converter);
  patterns.add<BuckyballInstLowering>(converter);
  patterns.add<BuckyballMvoverLowering>(converter);
}

void configureBaseLegalizeForExportTarget(LLVMConversionTarget &target) {
  target.addLegalOp<CustomIntrOp, MsetIntrOp, MvinIntrOp, MvinMmioIntrOp,
                    MvoutIntrOp>();
  target.addIllegalOp<MvoverOp, InstOp, MsetOp, MsetTransferOp, MvinOp,
                      Mvin2dOp, MvinMmioOp, MvoutOp>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp op) {
    return !contiguousRankOne(op.getSource().getType()) ||
           !contiguousRankOne(op.getTarget().getType());
  });
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
}

} // namespace legalize
} // namespace buckyball
} // namespace buddy
