class FP32ToRuntime : public OpRewritePattern<FP32MemMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FP32MemMatmulOp op, PatternRewriter &b) const override {
    auto a = cast<MemRefType>(op.getLhs().getType());
    auto w = cast<MemRefType>(op.getRhs().getType());
    auto out = cast<MemRefType>(op.getOutput().getType());
    int rank = a.getRank();
    if ((rank != 2 && rank != 3) || w.getRank() != rank || out.getRank() != rank ||
        !a.getElementType().isF32() || !w.getElementType().isF32() || !out.getElementType().isF32())
      return op.emitError("FP32 matmul requires rank-2 or rank-3 FP32 operands");
    buckyball_target::getBuckyballFunct7("SMATMUL_F32");
    auto type = UnrankedMemRefType::get(b.getF32Type(), 0);
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    if (!module.lookupSymbol<func::FuncOp>("fp32_matmul")) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto fn = b.create<func::FuncOp>(loc, "fp32_matmul", b.getFunctionType({type, type, type}, {}));
      fn.setPrivate();
      fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    }
    b.create<func::CallOp>(loc, "fp32_matmul", TypeRange{}, ValueRange{
      b.create<memref::CastOp>(loc, type, op.getLhs()),
      b.create<memref::CastOp>(loc, type, op.getRhs()),
      b.create<memref::CastOp>(loc, type, op.getOutput())});
    b.eraseOp(op);
    return success();
  }
};
