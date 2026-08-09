//===- ToyLowerBuckyballToLLVMPatterns.cpp - Toy LLVM patterns ------------===//

#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
  auto i32 = IntegerType::get(context, 32);
  auto ptr = LLVM::LLVMPointerType::get(context);
  return LLVM::LLVMFunctionType::get(i32, ptr, true);
}

static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                    getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global =
        builder.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal,
                                       name, builder.getStringAttr(value), 0);
  }

  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value zero = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getI64IntegerAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({zero, zero}));
}

class BBPrintMemRefOpLowering : public ConversionPattern {
public:
  explicit BBPrintMemRefOpLowering(MLIRContext *context)
      : ConversionPattern(PrintMemRefOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = cast<MemRefType>(*op->operand_type_begin());
    auto shape = memRefType.getShape();
    Type elemTy = memRefType.getElementType();
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, module);

    Value fmt;
    if (elemTy == rewriter.getF32Type() || elemTy == rewriter.getF64Type()) {
      fmt = getOrCreateGlobalString(loc, rewriter, "frmt_spec",
                                    StringRef("%f \0", 4), module);
    } else if (elemTy == rewriter.getI8Type() ||
               elemTy == rewriter.getI32Type()) {
      fmt = getOrCreateGlobalString(loc, rewriter, "frmt_spec",
                                    StringRef("%d \0", 4), module);
    } else {
      op->emitError("print_memref supports only i8, i32, f32 and f64");
      return failure();
    }

    Value nl = getOrCreateGlobalString(loc, rewriter, "nl",
                                       StringRef("\n\0", 2), module);
    SmallVector<Value, 4> ivs;
    for (unsigned i = 0, e = shape.size(); i != e; ++i) {
      auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto ub = rewriter.create<arith::ConstantIndexOp>(loc, shape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      for (Operation &nested : llvm::make_early_inc_range(*loop.getBody()))
        rewriter.eraseOp(&nested);
      ivs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());
      if (i != e - 1)
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(rewriter.getContext()),
                                      printfRef, nl);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    auto printOp = cast<PrintMemRefOp>(op);
    Value elem = rewriter.create<memref::LoadOp>(loc, printOp.getInput(), ivs);
    if (elem.getType() == rewriter.getF32Type())
      elem = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), elem);
    else if (elem.getType() == rewriter.getI8Type())
      elem = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI32Type(), elem);

    rewriter.create<LLVM::CallOp>(loc, getPrintfType(rewriter.getContext()),
                                  printfRef, ArrayRef<Value>({fmt, elem}));
    rewriter.eraseOp(op);
    return success();
  }
};

class BBPrintScalarOpLowering : public ConversionPattern {
public:
  explicit BBPrintScalarOpLowering(MLIRContext *context)
      : ConversionPattern(PrintScalarOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, module);
    Type elemTy = op->getOperand(0).getType();

    Value fmt;
    if (elemTy == rewriter.getF32Type() || elemTy == rewriter.getF64Type()) {
      fmt = getOrCreateGlobalString(loc, rewriter, "scalar_fmt",
                                    StringRef("%f\n\0", 5), module);
    } else if (elemTy == rewriter.getI8Type() ||
               elemTy == rewriter.getI32Type()) {
      fmt = getOrCreateGlobalString(loc, rewriter, "scalar_fmt",
                                    StringRef("%d\n\0", 5), module);
    } else {
      op->emitError("print_scalar supports only i8, i32, f32 and f64");
      return failure();
    }

    Value value = op->getOperand(0);
    if (elemTy == rewriter.getF32Type())
      value = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), value);
    else if (elemTy == rewriter.getI8Type())
      value = rewriter.create<LLVM::SExtOp>(loc, rewriter.getI32Type(), value);

    rewriter.create<LLVM::CallOp>(loc, getPrintfType(rewriter.getContext()),
                                  printfRef, ArrayRef<Value>({fmt, value}));
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateToyLowerBuckyballToLLVMPatterns(
    RewritePatternSet &patterns) {
  patterns.add<BBPrintMemRefOpLowering, BBPrintScalarOpLowering>(
      patterns.getContext());
}

void mlir::buddy::configureToyLowerBuckyballToLLVMTarget(
    LLVMConversionTarget &target) {
  target.addIllegalOp<PrintMemRefOp, PrintScalarOp>();
}
