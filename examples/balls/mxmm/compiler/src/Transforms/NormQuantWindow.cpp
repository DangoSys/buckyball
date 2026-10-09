#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
Value origin(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(
            op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

bool privateUses(Value value, ArrayRef<Operation *> allowed,
                 SmallVectorImpl<Operation *> &views) {
  llvm::SmallPtrSet<Operation *, 16> seen;
  std::function<bool(Value)> visit = [&](Value current) {
    for (Operation *user : current.getUsers()) {
      if (llvm::is_contained(allowed, user))
        continue;
      if (!isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(
              user))
        return false;
      if (seen.insert(user).second) {
        if (!visit(user->getResult(0)))
          return false;
        views.push_back(user);
      }
    }
    return true;
  };
  return visit(value);
}

SmallVector<Value> matrix(PatternRewriter &b, MXFP8MemMatmulOp op, Value r,
                          Value w, Value mean, Value epsilon) {
  Location loc = op.getLoc();
  int64_t width = op.getReductionK(), chunk = op.getTileK();
  int64_t columns = op.getTileN(), bytes = op.getBankBytes();
  auto outputType = cast<MemRefType>(op.getOutput().getType());
  int64_t outputWidth = outputType.getDimSize(1);
  int64_t chunks = (width + chunk - 1) / chunk;
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value four = b.create<arith::ConstantIndexOp>(loc, 4);
  Value limit = b.create<arith::ConstantIndexOp>(loc, outputWidth);
  Value step = b.create<arith::ConstantIndexOp>(loc, columns);
  auto loop = b.create<scf::ForOp>(loc, zero, limit, step, ValueRange{r, w});
  b.setInsertionPointToStart(loop.getBody());
  r = loop.getRegionIterArgs()[0];
  w = loop.getRegionIterArgs()[1];
  Value n = loop.getInductionVar();
  Value panel = b.create<arith::DivUIOp>(loc, n, step);
  for (int64_t part = 0; part < chunks; ++part) {
    int64_t count = std::min(chunk, width - part * chunk);
    Value offset = b.create<arith::MulIOp>(
        loc,
        b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, panel, b.create<arith::ConstantIndexOp>(loc, chunks)),
            b.create<arith::ConstantIndexOp>(loc, part)),
        b.create<arith::ConstantIndexOp>(loc, bytes));
    auto slice = b.create<memref::SubViewOp>(
        loc, op.getWeight(), SmallVector<OpFoldResult>{offset},
        SmallVector<OpFoldResult>{b.getIndexAttr(bytes)},
        SmallVector<OpFoldResult>{b.getIndexAttr(1)});
    Value view = b.create<memref::ExpandShapeOp>(
        loc, ArrayRef<int64_t>{bytes / 16, 16}, slice,
        ArrayRef<ReassociationIndices>{{0, 1}});
    r = b.create<BankMvinOp>(
        loc, b.getI64Type(), view, r,
        createI64Const(b, loc, columns * count * 33 / 32 / 16),
        createI64Const(b, loc, 1), b.getI64IntegerAttr(0));
    auto window = b.create<BankKernelOp>(
        loc, b.getI64Type(), b.getI64Type(), r, w,
        FlatSymbolRefAttr::get(b.getContext(), "rvv_norm_window"),
        ValueRange{createI64Const(b, loc, 1), createI64Const(b, loc, width),
                   createI64Const(b, loc, columns),
                   createI64Const(b, loc, count), createI64Const(b, loc, width),
                   createI64Const(b, loc, part * chunk),
                   createI64Const(b, loc, part == 0),
                   createI64Const(b, loc, part == chunks - 1),
                   createI64Const(b, loc, bytes), mean, epsilon});
    r = window.getReadOut();
    w = window.getWriteOut();
  }
  Value end = b.create<arith::MinUIOp>(loc, step,
                                       b.create<arith::SubIOp>(loc, limit, n));
  Value rows = b.create<arith::DivUIOp>(loc, end, four);
  SmallVector<OpFoldResult> offsets{b.getIndexAttr(0), n};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), end};
  SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
  auto sliceType = memref::SubViewOp::inferRankReducedResultType(
      {ShapedType::kDynamic}, outputType, offsets, sizes, strides);
  Value destination = b.create<memref::SubViewOp>(
      loc, sliceType, op.getOutput(), offsets, sizes, strides);
  Value dmaView = b.create<memref::ExpandShapeOp>(
      loc, ArrayRef<int64_t>{ShapedType::kDynamic, 4}, destination,
      ArrayRef<ReassociationIndices>{{0, 1}},
      SmallVector<OpFoldResult>{rows, b.getIndexAttr(4)});
  w = b.create<BankMvoutOp>(
      loc, b.getI64Type(), dmaView, w,
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), rows),
      createI64Const(b, loc, 1), b.getI64IntegerAttr(1));
  b.create<FenceOp>(loc);
  b.create<scf::YieldOp>(loc, ValueRange{r, w});
  b.setInsertionPointAfter(loop);
  b.eraseOp(op);
  return {loop.getResult(0), loop.getResult(1)};
}

class FuseNormQuantWindow : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::CallOp quant,
                                PatternRewriter &b) const override {
    if (quant.getCallee() != "mxfp8_quant" || quant.getNumOperands() != 5)
      return failure();
    Value normBuffer = origin(quant.getOperand(0));
    Value packedBuffer = origin(quant.getOperand(1));
    auto normType = dyn_cast<MemRefType>(normBuffer.getType());
    auto packedType = dyn_cast<MemRefType>(packedBuffer.getType());
    if (!normBuffer.getDefiningOp<memref::AllocOp>() ||
        !packedBuffer.getDefiningOp<memref::AllocOp>() || !normType ||
        !packedType || !normType.hasStaticShape() || normType.getRank() < 1 ||
        !packedType.hasStaticShape() || !normType.getElementType().isF32() ||
        !packedType.getElementType().isInteger(8))
      return failure();
    func::CallOp norm;
    for (auto call : quant->getBlock()->getOps<func::CallOp>())
      if (call.getCallee() == "rvv_norm" && call.getNumOperands() == 5 &&
          origin(call.getOperand(0)) == normBuffer &&
          call->isBeforeInBlock(quant)) {
        if (norm)
          return failure();
        norm = call;
      }
    if (!norm)
      return failure();
    SmallVector<MXFP8MemMatmulOp> matrices;
    for (auto op : quant->getBlock()->getOps<MXFP8MemMatmulOp>())
      if (origin(op.getInput()) == packedBuffer)
        matrices.push_back(op);
    if (matrices.empty() || matrices.size() > 2)
      return failure();
    if (matrices.size() == 2 && !matrices[0]->isBeforeInBlock(matrices[1]))
      std::swap(matrices[0], matrices[1]);
    const auto &target = buckyball_target::getBuckyballTarget();
    if (!target.rvvEnabled || target.bankWidthBits != 128 ||
        target.bankNum < 5 ||
        !llvm::is_contained(target.balls, "MxquantBall") ||
        !llvm::any_of(
            target.isa,
            [](const auto &entry) {
              return entry.mnemonic == "MXMM_MXFP8_WINDOW";
            }))
      return failure();
    auto module = quant->getParentOfType<ModuleOp>();
    auto normFunction = module.lookupSymbol<func::FuncOp>("rvv_norm");
    auto quantFunction = module.lookupSymbol<func::FuncOp>("mxfp8_quant");
    if (!normFunction || !normFunction.isExternal() || !quantFunction ||
        !quantFunction.isExternal())
      return failure();
    int64_t width = matrices[0].getReductionK(), chunk = matrices[0].getTileK();
    int64_t activationChunk = matrices[0].getActivationTileK();
    int64_t bytes = target.bankDepth * 16;
    if (chunk <= 0 || chunk % 32 || chunk >= 4096 || width <= 0 || width % 32 ||
        (activationChunk != chunk && activationChunk != width))
      return failure();
    int64_t chunks = (width + chunk - 1) / chunk;
    if (width <= 0 || width % 32 || width * 4 > bytes ||
        width + width / 32 > bytes || normType.getNumElements() != width ||
        normType.getDimSize(normType.getRank() - 1) != width ||
        packedType.getRank() != 1 ||
        packedType.getNumElements() !=
            ((width + activationChunk - 1) / activationChunk) * bytes)
      return failure();
    const int64_t quantParameters[] = {1, activationChunk, bytes};
    for (size_t index = 0; index < 3; ++index) {
      auto constant =
          quant.getOperand(index + 2).getDefiningOp<arith::ConstantOp>();
      if (!constant || cast<IntegerAttr>(constant.getValue()).getInt() !=
                           quantParameters[index])
        return failure();
    }
    for (Operation *cursor = norm.getOperation();
         cursor != matrices.back().getOperation()->getNextNode();
         cursor = cursor->getNextNode()) {
      if (cursor == norm || cursor == quant || cursor == matrices[0] ||
          cursor == matrices.back())
        continue;
      if (isa<memref::AllocOp>(cursor))
        continue;
      if (isa<func::CallOp>(cursor) || cursor->getNumRegions() ||
          !isMemoryEffectFree(cursor))
        return failure();
    }
    for (auto op : matrices) {
      auto output = cast<MemRefType>(op.getOutput().getType());
      auto weight = cast<MemRefType>(op.getWeight().getType());
      int64_t columns = op.getTileN();
      if (!output.hasStaticShape() || output.getRank() != 2 ||
          output.getDimSize(0) != 1 || output.getDimSize(1) % 4 ||
          output.getStridesAndOffset().first.back() != 1 ||
          !op.getOutput().getDefiningOp<memref::AllocOp>() || columns <= 0 ||
          columns % 16 || op.getReductionK() != width ||
          op.getTileK() != chunk ||
          op.getActivationTileK() != matrices[0].getActivationTileK() ||
          op.getBankBytes() != bytes || columns * chunk * 33 / 32 > bytes ||
          columns * 4 > bytes || weight.getRank() != 1 ||
          weight.getStridesAndOffset().first[0] != 1 ||
          weight.getDimSize(0) !=
              ((output.getDimSize(1) + columns - 1) / columns) * chunks *
                  bytes ||
          !quant->isBeforeInBlock(op))
        return failure();
    }
    SmallVector<Operation *> views;
    SmallVector<Operation *> packedUsers{quant.getOperation()};
    for (auto matrix : matrices)
      packedUsers.push_back(matrix.getOperation());
    if (!privateUses(normBuffer, {norm.getOperation(), quant.getOperation()},
                     views) ||
        !privateUses(packedBuffer, packedUsers, views))
      return failure();
    Value input = norm.getOperand(1), gamma = norm.getOperand(2);
    for (Value *value : {&input, &gamma})
      while (auto cast = value->getDefiningOp<memref::CastOp>())
        *value = cast.getSource();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || inputType.getRank() < 1 || !inputType.hasStaticShape() ||
        inputType.getDimSize(inputType.getRank() - 1) != width ||
        origin(input) == normBuffer || origin(gamma) == normBuffer ||
        origin(input) == packedBuffer || origin(gamma) == packedBuffer)
      return failure();
    for (Value value : {input, gamma}) {
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || !type.hasStaticShape() || type.getNumElements() != width ||
          !type.getElementType().isF32())
        return failure();
      auto strides = type.getStridesAndOffset().first;
      int64_t stride = 1;
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        if (type.getDimSize(axis) != 1 && strides[axis] != stride)
          return failure();
        stride *= type.getDimSize(axis);
      }
    }
    Location loc = norm.getLoc();
    b.setInsertionPoint(norm);
    Value d = allocBank(b, loc, 1, 1), n = allocBank(b, loc, 1, 1),
          x = allocBank(b, loc, 1, 1), g = allocBank(b, loc, 1, 1),
          a = allocBank(b, loc, 1, 1);
    auto rowView = [&](Value value) {
      auto type = cast<MemRefType>(value.getType());
      ReassociationIndices axes;
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        axes.push_back(axis);
      if (type.getRank() != 1)
        value = b.create<memref::CollapseShapeOp>(
            loc, value, ArrayRef<ReassociationIndices>{axes});
      return Value(b.create<memref::ExpandShapeOp>(
          loc, ArrayRef<int64_t>{width / 4, 4}, value,
          ArrayRef<ReassociationIndices>{{0, 1}}));
    };
    x = mvinBank(b, loc, rowView(input), x, width / 4);
    g = mvinBank(b, loc, rowView(gamma), g, width / 4);
    Value r = b.create<BankTransferOp>(loc, b.getI64Type(), g, x);
    Value w = b.create<BankTransferOp>(loc, b.getI64Type(), n, d);
    w = b.create<BankTransferOp>(loc, b.getI64Type(), a, w);
    auto initialized = b.create<BankKernelOp>(
        loc, b.getI64Type(), b.getI64Type(), r, w,
        FlatSymbolRefAttr::get(b.getContext(), "rvv_norm_window"),
        ValueRange{createI64Const(b, loc, 0), createI64Const(b, loc, width),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   createI64Const(b, loc, width), createI64Const(b, loc, 0),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   createI64Const(b, loc, bytes), norm.getOperand(3),
                   norm.getOperand(4)});
    Value mean = norm.getOperand(3), epsilon = norm.getOperand(4);
    SmallVector<Value> state{initialized.getReadOut(),
                             initialized.getWriteOut()};
    b.eraseOp(norm);
    b.eraseOp(quant);
    for (auto op : matrices) {
      b.setInsertionPoint(op);
      state = matrix(b, op, state[0], state[1], mean, epsilon);
    }
    auto finished = b.create<BankKernelOp>(
        loc, b.getI64Type(), b.getI64Type(), state[0], state[1],
        FlatSymbolRefAttr::get(b.getContext(), "rvv_norm_window"),
        ValueRange{createI64Const(b, loc, 2), createI64Const(b, loc, width),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   createI64Const(b, loc, width), createI64Const(b, loc, 0),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   createI64Const(b, loc, bytes), mean, epsilon});
    b.create<FenceOp>(loc);
    releaseBank(b, loc, finished.getReadOut());
    releaseBank(b, loc, finished.getWriteOut());
    for (Operation *view : views)
      b.eraseOp(view);
    b.eraseOp(normBuffer.getDefiningOp());
    b.eraseOp(packedBuffer.getDefiningOp());
    return success();
  }
};

class NormQuantWindowPass
    : public PassWrapper<NormQuantWindowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormQuantWindowPass)
  StringRef getArgument() const final { return "fuse-norm-quant-window"; }
  StringRef getDescription() const final {
    return "Keep full-row Norm and quantized activation in banks across paired "
           "matrix consumers.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect, BuckyballDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseNormQuantWindow>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
namespace mlir::buddy {
void registerNormQuantWindowPass() { PassRegistration<NormQuantWindowPass>(); }
} // namespace mlir::buddy
