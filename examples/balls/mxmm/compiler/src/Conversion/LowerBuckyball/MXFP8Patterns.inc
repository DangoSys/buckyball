class MXFP8ToBanks : public OpRewritePattern<MXFP8MemMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MXFP8MemMatmulOp op,
                                PatternRewriter &b) const override {
    auto aTy = cast<MemRefType>(op.getInput().getType());
    auto wTy = cast<MemRefType>(op.getWeight().getType());
    auto oTy = cast<MemRefType>(op.getOutput().getType());
    if (!aTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !oTy.hasStaticShape() || aTy.getRank() != 1 || wTy.getRank() != 1 ||
        oTy.getRank() != 2 || !aTy.getElementType().isInteger(8) ||
        !wTy.getElementType().isInteger(8) || !oTy.getElementType().isF32())
      return op.emitError(
          "MXFP8 requires static packed activation/weight bytes "
          "and an FP32 output matrix");
    if (aTy.getStridesAndOffset().first[0] != 1 ||
        wTy.getStridesAndOffset().first[0] != 1)
      return op.emitError("MXFP8 packed tensors must have unit byte stride");
    int64_t M = oTy.getShape()[0], K = op.getReductionK(),
            N = oTy.getShape()[1];
    int64_t rows = M == 1 ? 1 : op.getTileM(), columns = op.getTileN();
    int64_t chunk = op.getTileK();
    if (rows <= 0 || (rows != 1 && rows % 16) || columns <= 0 || columns % 16 ||
        chunk <= 0 || chunk % 32 || chunk >= 4096)
      return op.emitError(
          "MXFP8 tile_k must be a positive multiple of 32 below 4096");
    int64_t chunks = (K + chunk - 1) / chunk;
    int64_t stride = op.getBankBytes();
    if (M <= 0 || K <= 0 || N <= 0 || K % 32 ||
        aTy.getShape()[0] != ((M + rows - 1) / rows) * chunks * stride ||
        wTy.getShape()[0] != ((N + columns - 1) / columns) * chunks * stride)
      return op.emitError("MXFP8 matrix or packed weight shape mismatch");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth * 16 != stride ||
        (rows != 1 && (rows <= 0 || rows % 16)) || columns <= 0 ||
        columns % 16 || (rows * chunk * 33 / 32 + 15) / 16 * 16 > stride ||
        columns * chunk * 33 / 32 > stride || rows * columns * 4 > stride)
      return op.emitError(
          "MXFP8 packed K block exceeds the 128-bit bank capacity");
    Location loc = op.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0),
          one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value packed = op.getInput();
    auto dynamic = ShapedType::kDynamic;
    auto bankView = [&](Value source, Value offset, int64_t size) -> Value {
      Value slice = b.create<memref::SubViewOp>(
          loc, source, SmallVector<OpFoldResult>{offset},
          SmallVector<OpFoldResult>{b.getIndexAttr(stride)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1)});
      auto ty = MemRefType::get(
          {stride / 16, 16}, b.getI8Type(),
          StridedLayoutAttr::get(b.getContext(), dynamic, {16, 1}));
      return b.create<memref::ExpandShapeOp>(
          loc, ty, slice, ArrayRef<ReassociationIndices>{{0, 1}});
    };
    Value aBank = allocBank(b, loc, 1, 1), wBank = allocBank(b, loc, 1, 1),
          cBank = allocBank(b, loc, 1, 1);
    bool residentInput = chunks + 2 <= target.bankNum;
    SmallVector<Value> banks{aBank};
    if (residentInput)
      for (int64_t part = 1; part < chunks; ++part)
        banks.push_back(allocBank(b, loc, 1, 1));
    banks.push_back(wBank);
    banks.push_back(cBank);
    auto outPackTy = MemRefType::get({stride / 16, 4}, b.getF32Type());
    Value outPack = b.create<memref::AllocOp>(loc, outPackTy);
    outPack.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(stride));
    Value limit = b.create<arith::ConstantIndexOp>(loc, N),
          step = b.create<arith::ConstantIndexOp>(loc, columns);
    Value rowLimit = b.create<arith::ConstantIndexOp>(loc, M),
          rowStep = b.create<arith::ConstantIndexOp>(loc, rows);
    auto rowBlocks = b.create<scf::ForOp>(loc, zero, rowLimit, rowStep, banks);
    b.setInsertionPointToStart(rowBlocks.getBody());
    Value m = rowBlocks.getInductionVar();
    SmallVector<Value> inputBanks(rowBlocks.getRegionIterArgs().drop_back(2));
    aBank = inputBanks.front();
    wBank = rowBlocks.getRegionIterArgs()[banks.size() - 2];
    cBank = rowBlocks.getRegionIterArgs()[banks.size() - 1];
    Value rowPanel = b.create<arith::DivUIOp>(loc, m, rowStep);
    if (residentInput) {
      for (int64_t part = 0; part < chunks; ++part) {
        Value offset = b.create<arith::MulIOp>(
            loc,
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, rowPanel,
                    b.create<arith::ConstantIndexOp>(loc, chunks)),
                b.create<arith::ConstantIndexOp>(loc, part)),
            b.create<arith::ConstantIndexOp>(loc, stride));
        int64_t count = std::min(chunk, K - part * chunk);
        int64_t aBytes = (rows * count * 33 / 32 + 15) / 16 * 16;
        inputBanks[part] = mvinBank(b, loc, bankView(packed, offset, stride),
                                    inputBanks[part], aBytes / 16);
      }
      aBank = inputBanks.front();
    }
    auto panels = b.create<scf::ForOp>(loc, zero, limit, step,
                                       ValueRange{aBank, wBank, cBank});
    b.setInsertionPointToStart(panels.getBody());
    Value n = panels.getInductionVar();
    aBank = panels.getRegionIterArgs()[0];
    wBank = panels.getRegionIterArgs()[1];
    cBank = panels.getRegionIterArgs()[2];
    Value state = cBank;
    for (int64_t part = 0; part < chunks; ++part) {
      int64_t count = std::min(chunk, K - part * chunk);
      int64_t aBytes = (rows * count * 33 / 32 + 15) / 16 * 16,
              bBytes = columns * count * 33 / 32;
      Value aOffset = b.create<arith::MulIOp>(
          loc,
          b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, rowPanel, b.create<arith::ConstantIndexOp>(loc, chunks)),
              b.create<arith::ConstantIndexOp>(loc, part)),
          b.create<arith::ConstantIndexOp>(loc, stride));
      Value panel = b.create<arith::DivUIOp>(loc, n, step);
      Value wOffset = b.create<arith::MulIOp>(
          loc,
          b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, panel, b.create<arith::ConstantIndexOp>(loc, chunks)),
              b.create<arith::ConstantIndexOp>(loc, part)),
          b.create<arith::ConstantIndexOp>(loc, stride));
      if (!residentInput)
        aBank = mvinBank(b, loc, bankView(packed, aOffset, aBytes), aBank,
                         aBytes / 16);
      wBank = mvinBank(b, loc, bankView(op.getWeight(), wOffset, bBytes), wBank,
                       bBytes / 16);
      state = b.create<BankMXFP8Op>(
          loc, b.getI64Type(), residentInput ? inputBanks[part] : aBank, wBank,
          state, createI64Const(b, loc, matrixRs2(rows, columns, count)),
          createI1Const(b, loc, part == 0),
          createI1Const(b, loc, part == chunks - 1), createI64Const(b, loc, 0));
    }
    cBank = mvoutBank(b, loc, outPack, state, rows * columns / 4);
    b.create<FenceOp>(loc);
    Value end = b.create<arith::MinUIOp>(
        loc, step, b.create<arith::SubIOp>(loc, limit, n));
    SmallVector<int64_t> outputStrides;
    int64_t outputOffset;
    if (failed(oTy.getStridesAndOffset(outputStrides, outputOffset)))
      return op.emitError("MXFP8 matmul requires a strided output");
    auto rr = b.create<scf::ForOp>(
        loc, zero,
        b.create<arith::MinUIOp>(loc, rowStep,
                                 b.create<arith::SubIOp>(loc, rowLimit, m)),
        one);
    b.setInsertionPointToStart(rr.getBody());
    Value r = rr.getInductionVar();
    if (outputStrides.back() == 1) {
      auto flatType = MemRefType::get({stride / 4}, b.getF32Type());
      Value flat = b.create<memref::CollapseShapeOp>(
          loc, flatType, outPack, ArrayRef<ReassociationIndices>{{0, 1}});
      SmallVector<OpFoldResult> offsets{
          b.create<arith::MulIOp>(
               loc, r, b.create<arith::ConstantIndexOp>(loc, columns))
              .getResult()};
      SmallVector<OpFoldResult> sizes{end}, steps{b.getIndexAttr(1)};
      auto sourceType = memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic}, flatType, offsets, sizes, steps);
      Value source = b.create<memref::SubViewOp>(loc, sourceType, flat, offsets,
                                                 sizes, steps);
      SmallVector<OpFoldResult> outOffsets{
          b.create<arith::AddIOp>(loc, m, r).getResult(), n};
      SmallVector<OpFoldResult> outSizes{b.getIndexAttr(1), end};
      SmallVector<OpFoldResult> outSteps(2, b.getIndexAttr(1));
      auto rowType = memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic}, oTy, outOffsets, outSizes, outSteps);
      Value destination = b.create<memref::SubViewOp>(
          loc, rowType, op.getOutput(), outOffsets, outSizes, outSteps);
      b.create<memref::CopyOp>(loc, source, destination);
    } else {
      auto cc = b.create<scf::ForOp>(loc, zero, end, one);
      b.setInsertionPointToStart(cc.getBody());
      Value c = cc.getInductionVar(),
            four = b.create<arith::ConstantIndexOp>(loc, 4);
      Value pr = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, r, b.create<arith::ConstantIndexOp>(loc, columns / 4)),
          b.create<arith::DivUIOp>(loc, c, four));
      Value pc = b.create<arith::RemUIOp>(loc, c, four);
      Value value = b.create<memref::LoadOp>(loc, outPack, ValueRange{pr, pc});
      b.create<memref::StoreOp>(loc, value, op.getOutput(),
                                ValueRange{b.create<arith::AddIOp>(loc, r, m),
                                           b.create<arith::AddIOp>(loc, n, c)});
    }
    b.setInsertionPointAfter(rr);
    b.create<scf::YieldOp>(loc, ValueRange{aBank, wBank, cBank});
    b.setInsertionPointAfter(panels);
    inputBanks[0] = panels.getResult(0);
    inputBanks.push_back(panels.getResult(1));
    inputBanks.push_back(panels.getResult(2));
    b.create<scf::YieldOp>(loc, inputBanks);
    b.setInsertionPointAfter(rowBlocks);
    for (Value bank : rowBlocks.getResults())
      releaseBank(b, loc, bank);
    b.create<memref::DeallocOp>(loc, outPack);
    b.eraseOp(op);
    return success();
  }
};
