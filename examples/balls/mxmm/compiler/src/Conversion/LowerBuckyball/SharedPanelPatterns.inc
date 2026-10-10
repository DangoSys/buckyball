class MXFP8SharedPanelToBanks : public OpRewritePattern<MXFP8SharedPanelOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MXFP8SharedPanelOp op,
                                PatternRewriter &b) const override {
    auto a = cast<MemRefType>(op.getInput().getType());
    auto w = cast<MemRefType>(op.getWeight().getType());
    int64_t rows = op.getRows(), columns = op.getColumns();
    int64_t k = op.getReductionK(), chunk = op.getTileK();
    int64_t bytes = op.getBankBytes();
    int64_t sharedBytes = op.getSharedBytesAttr().getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    if ((rows != 1 && (rows <= 0 || rows % 16)) || columns <= 0 ||
        columns % 16 || rows > 4095 || columns > 4095 || k <= 0 || k % 32 ||
        chunk <= 0 || chunk % 32 || chunk >= 4096 ||
        target.bankWidthBits != 128 || target.bankDepth * 16 != bytes ||
        sharedBytes <= 0 || sharedBytes % 16 ||
        rows * columns * 4 > sharedBytes ||
        (rows * chunk * 33 / 32 + 15) / 16 * 16 > bytes ||
        columns * chunk * 33 / 32 > bytes)
      return op.emitError(
          "shared MXFP8 panel exceeds the original bank layout");
    int64_t chunks = (k + chunk - 1) / chunk;
    for (MemRefType type : {a, w})
      if (!type.hasStaticShape() || type.getRank() != 1 ||
          !type.getElementType().isInteger(8) ||
          type.getShape()[0] != chunks * bytes ||
          type.getStridesAndOffset().first[0] != 1)
        return op.emitError(
            "shared MXFP8 panel requires one packed row/N panel");
    Location loc = op.getLoc();
    Value ab = allocBank(b, loc, 1, 1), wb = allocBank(b, loc, 1, 1);
    Value output = op.getBank();
    for (int64_t part = 0; part < chunks; ++part) {
      int64_t count = std::min(chunk, k - part * chunk);
      auto view = [&](Value packed) -> Value {
        Value slice = b.create<memref::SubViewOp>(
            loc, packed,
            SmallVector<OpFoldResult>{b.getIndexAttr(part * bytes)},
            SmallVector<OpFoldResult>{b.getIndexAttr(bytes)},
            SmallVector<OpFoldResult>{b.getIndexAttr(1)});
        auto type = *memref::ExpandShapeOp::computeExpandedType(
            cast<MemRefType>(slice.getType()), {bytes / 16, 16},
            ArrayRef<ReassociationIndices>{{0, 1}});
        return b.create<memref::ExpandShapeOp>(
            loc, type, slice, ArrayRef<ReassociationIndices>{{0, 1}});
      };
      ab = mvinBank(b, loc, view(op.getInput()), ab,
                    (rows * count * 33 / 32 + 15) / 16);
      wb = mvinBank(b, loc, view(op.getWeight()), wb,
                    columns * count * 33 / 32 / 16);
      output = b.create<BankMXFP8Op>(
          loc, b.getI64Type(), ab, wb, output,
          createI64Const(b, loc, matrixRs2(rows, columns, count)),
          createI1Const(b, loc, part == 0),
          createI1Const(b, loc, part == chunks - 1), createI64Const(b, loc, 0));
    }
    releaseBank(b, loc, ab);
    releaseBank(b, loc, wb);
    b.replaceOp(op, output);
    return success();
  }
};
