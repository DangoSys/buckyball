struct MXFP8WindowLowering : public ConvertOpToLLVMPattern<MXFP8WindowOp> {
  MXFP8WindowLowering(LLVMTypeConverter &converter, bool, int64_t bankDepth)
      : ConvertOpToLLVMPattern<MXFP8WindowOp>(converter), depth(bankDepth) {}

  LogicalResult matchAndRewrite(MXFP8WindowOp op, OpAdaptor a,
                                ConversionPatternRewriter &b) const override {
    buckyball_target::requireBuckyballBall("MxmmBall");
    if (buckyball_target::getBuckyballTarget().bankWidthBits != 128)
      return op.emitError("mxmm window: requires 16-byte bank rows");
    auto integer = [](Value value) -> std::optional<int64_t> {
      if (auto constant = value.getDefiningOp<arith::ConstantOp>())
        return cast<IntegerAttr>(constant.getValue()).getInt();
      return std::nullopt;
    };
    auto rows = integer(op.getRows()), cols = integer(op.getCols());
    auto count = integer(op.getCount()), full = integer(op.getFullK());
    auto start = integer(op.getStartK()), base = integer(op.getOutputBase());
    if (rows && (*rows > 4095 || (*rows != 1 && (*rows <= 0 || *rows % 16))))
      return op.emitError("mxmm window: rows must be 1 or a positive multiple "
                          "of 16 within 12 bits");
    if (cols && (*cols <= 0 || *cols > 4095 || *cols % 16))
      return op.emitError(
          "mxmm window: cols must be a positive multiple of 16 within 12 bits");
    if (count &&
        (*count <= 0 || uint64_t(*count) >= (uint64_t(1) << 34) || *count % 32))
      return op.emitError("mxmm window: count must be a positive multiple of "
                          "32 within 34 bits");
    if (full && (*full <= 0 || *full > 65535 || *full % 32))
      return op.emitError("mxmm window: fullK must be a positive multiple of "
                          "32 within 16 bits");
    if (start && (*start < 0 || *start > 65535 || *start % 32))
      return op.emitError("mxmm window: startK must be a nonnegative multiple "
                          "of 32 within 16 bits");
    if (base && (*base < 0 || *base > 63))
      return op.emitError("mxmm window: outputBase must fit 6 bits");
    if (full && start && count &&
        uint64_t(*start) + uint64_t(*count) > uint64_t(*full))
      return op.emitError(
          "mxmm window: reduction exceeds complete activation row");
    if (rows && full &&
        uint64_t(*rows) * uint64_t(*full) * 33 / 32 > uint64_t(depth) * 16)
      return op.emitError(
          "mxmm window: complete activation exceeds bank capacity");
    if (cols && count &&
        uint64_t(*cols) * uint64_t(*count) * 33 / 32 > uint64_t(depth) * 16)
      return op.emitError("mxmm window: weight exceeds bank capacity");
    if (rows && cols && base &&
        uint64_t(*base) * 16 + uint64_t(*rows) * uint64_t(*cols) * 4 >
            uint64_t(depth) * 16)
      return op.emitError("mxmm window: output exceeds bank capacity");

    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(b, loc, a.getOp1BankId(), a.getOp2BankId(),
                                 a.getResultBankId(), a.getCount());
    Value rs2 = b.create<arith::AndIOp>(loc, a.getRows(), cstI64(b, loc, 4095));
    SmallVector<Value> fields{a.getCols(),       a.getFirst(), a.getLast(),
                              a.getOutputBase(), a.getFullK(), a.getStartK()};
    const int64_t shifts[] = {12, 24, 25, 26, 32, 48};
    const int64_t masks[] = {4095, 1, 1, 63, 65535, 65535};
    for (size_t index = 0; index < fields.size(); ++index) {
      Value value = fields[index];
      if (value.getType().isInteger(1))
        value = b.create<arith::ExtUIOp>(loc, b.getI64Type(), value);
      value = b.create<arith::AndIOp>(loc, value, cstI64(b, loc, masks[index]));
      rs2 = b.create<arith::OrIOp>(
          loc, rs2,
          b.create<arith::ShLIOp>(loc, value, cstI64(b, loc, shifts[index])));
    }
    b.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        b.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("MXMM_MXFP8_WINDOW")));
    return success();
  }
  int64_t depth;
};
