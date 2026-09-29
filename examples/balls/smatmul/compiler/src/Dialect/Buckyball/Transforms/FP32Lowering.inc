struct FP32Lowering : public ConvertOpToLLVMPattern<FP32Op> {
  FP32Lowering(LLVMTypeConverter &converter, bool, int64_t depth)
      : ConvertOpToLLVMPattern<FP32Op>(converter), depth(depth) {}

  LogicalResult matchAndRewrite(FP32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    buckyball_target::requireBuckyballBall("SMatMulBall");
    auto constant = op.getConfig().getDefiningOp<arith::ConstantOp>();
    auto shape = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!shape) return op.emitError("FP32 matmul requires a constant shape");
    uint64_t cfg = shape.getValue().getZExtValue();
    uint64_t rows = cfg & 0xfff, cols = (cfg >> 12) & 0xfff, k = (cfg >> 24) & 0xfff;
    if (cfg >> 36 || (rows != 1 && rows != 16) || cols != 16 || !k || k % 4 ||
        rows * k / 4 > uint64_t(depth) || cols * k / 4 > uint64_t(depth) ||
        buckyball_target::getBuckyballTarget().bankWidthBits != 128)
      return op.emitError("FP32 matmul shape does not fit the configured banks");
    auto baseOp = op.getOutputBase().getDefiningOp<arith::ConstantOp>();
    auto base = baseOp ? dyn_cast<IntegerAttr>(baseOp.getValue()) : IntegerAttr();
    if (!base || base.getInt() < 0 || base.getInt() > 63 || base.getInt() + rows * 4 > uint64_t(depth))
      return op.emitError("FP32 output base does not fit the configured bank");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(b, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
                               adaptor.getResultBankId(), cstI64(b, loc, k));
    Value rs2 = cstI64(b, loc, matrixCfg(rows, cols, false, false));
    Value first = b.create<arith::ExtUIOp>(loc, b.getI64Type(), adaptor.getFirst());
    Value last = b.create<arith::ExtUIOp>(loc, b.getI64Type(), adaptor.getLast());
    rs2 = b.create<arith::OrIOp>(loc, rs2, b.create<arith::ShLIOp>(loc, first, cstI64(b, loc, 24)));
    rs2 = b.create<arith::OrIOp>(loc, rs2, b.create<arith::ShLIOp>(loc, last, cstI64(b, loc, 25)));
    rs2 = b.create<arith::OrIOp>(loc, rs2, b.create<arith::ShLIOp>(loc, adaptor.getOutputBase(), cstI64(b, loc, 26)));
    b.replaceOpWithNewOp<CustomIntrOp>(op, rs1, rs2,
        b.getI32IntegerAttr(buckyball_target::getBuckyballFunct7("SMATMUL_F32")));
    return success();
  }
  int64_t depth;
};
