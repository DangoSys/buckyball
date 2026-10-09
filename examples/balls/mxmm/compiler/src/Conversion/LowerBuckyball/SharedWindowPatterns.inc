class MXFP8SharedWindowToBanks : public OpRewritePattern<MXFP8SharedWindowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MXFP8SharedWindowOp op,
                                PatternRewriter &b) const override {
    auto a = cast<MemRefType>(op.getInput().getType());
    auto w = cast<MemRefType>(op.getWeight().getType());
    int64_t rows = op.getRows(), columns = op.getColumns();
    int64_t k = op.getReductionK(), chunk = op.getTileK();
    int64_t bytes = op.getBankBytes();
    int64_t sharedBytes = op.getSharedBytesAttr().getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    if ((rows != 1 && (rows <= 0 || rows % 16)) || rows > 4095 ||
        columns <= 0 || columns > 4095 || columns % 16 || k <= 0 || k > 65535 ||
        k % 32 || chunk <= 0 || chunk >= 4096 || chunk % 32 ||
        target.bankWidthBits != 128 || target.bankDepth * 16 != bytes ||
        sharedBytes <= 0 || sharedBytes % 16 ||
        rows * columns * 4 > sharedBytes ||
        (rows * k * 33 / 32 + 15) / 16 * 16 > bytes ||
        columns * chunk * 33 / 32 > bytes)
      return op.emitError(
          "shared MXFP8 window exceeds its activation or output bank");
    int64_t chunks = (k + chunk - 1) / chunk;
    if (!a.hasStaticShape() || !w.hasStaticShape() ||
        a.getShape()[0] != bytes || w.getShape()[0] != chunks * bytes ||
        a.getStridesAndOffset().first[0] != 1 ||
        w.getStridesAndOffset().first[0] != 1)
      return op.emitError(
          "shared MXFP8 window requires a complete activation and one N panel");
    Location loc = op.getLoc();
    auto view = [&](Value source, int64_t offset) -> Value {
      Value slice = b.create<memref::SubViewOp>(
          loc, source, SmallVector<OpFoldResult>{b.getIndexAttr(offset)},
          SmallVector<OpFoldResult>{b.getIndexAttr(bytes)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1)});
      auto type = *memref::ExpandShapeOp::computeExpandedType(
          cast<MemRefType>(slice.getType()), {bytes / 16, 16},
          ArrayRef<ReassociationIndices>{{0, 1}});
      return b.create<memref::ExpandShapeOp>(
          loc, type, slice, ArrayRef<ReassociationIndices>{{0, 1}});
    };
    Value ab = allocBank(b, loc, 1, 1), wb = allocBank(b, loc, 1, 1);
    ab = mvinBank(b, loc, view(op.getInput(), 0), ab,
                  (rows * k * 33 / 32 + 15) / 16);
    Value output = op.getBank();
    for (int64_t part = 0; part < chunks; ++part) {
      int64_t count = std::min(chunk, k - part * chunk);
      wb = mvinBank(b, loc, view(op.getWeight(), part * bytes), wb,
                    columns * count * 33 / 32 / 16);
      output = b.create<BankMXFP8WindowOp>(
          loc, b.getI64Type(), ab, wb, output, createI64Const(b, loc, rows),
          createI64Const(b, loc, columns), createI64Const(b, loc, count),
          createI64Const(b, loc, k), createI64Const(b, loc, part * chunk),
          createI1Const(b, loc, part == 0),
          createI1Const(b, loc, part == chunks - 1), createI64Const(b, loc, 0));
    }
    releaseBank(b, loc, ab);
    releaseBank(b, loc, wb);
    b.create<FenceOp>(loc);
    b.replaceOp(op, output);
    return success();
  }
};
