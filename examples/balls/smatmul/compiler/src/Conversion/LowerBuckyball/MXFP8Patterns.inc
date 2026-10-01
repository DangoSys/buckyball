class MXFP8ToBanks : public OpRewritePattern<MXFP8MemMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MXFP8MemMatmulOp op,
                                PatternRewriter &b) const override {
    auto aTy = cast<MemRefType>(op.getInput().getType());
    auto wTy = cast<MemRefType>(op.getWeight().getType());
    auto oTy = cast<MemRefType>(op.getOutput().getType());
    if (!aTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !oTy.hasStaticShape() || aTy.getRank() != 2 || wTy.getRank() != 1 ||
        oTy.getRank() != 2 || !aTy.getElementType().isF32() ||
        !wTy.getElementType().isInteger(8) || !oTy.getElementType().isF32())
      return op.emitError("MXFP8 requires static FP32 input/output matrices "
                          "and packed byte weights");
    if (wTy.getStridesAndOffset().first[0] != 1)
      return op.emitError("MXFP8 packed weights must have unit byte stride");
    int64_t M = aTy.getShape()[0], K = aTy.getShape()[1], N = oTy.getShape()[1];
    int64_t rows = M == 1 ? 1 : 16, chunk = 512,
            chunks = (K + chunk - 1) / chunk;
    int64_t aBytes = rows * chunk * 33 / 32, bBytes = 16 * chunk * 33 / 32;
    if (M <= 0 || K <= 0 || N <= 0 || K % 32 || oTy.getShape()[0] != M ||
        wTy.getShape()[0] != ((N + 15) / 16) * chunks * bBytes)
      return op.emitError("MXFP8 matrix or packed weight shape mismatch");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth * 16 < bBytes)
      return op.emitError(
          "MXFP8 requires 128-bit banks with at least 528 rows");
    Location loc = op.getLoc();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0),
          one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto packedTy = MemRefType::get({((M + rows - 1) / rows) * chunks * aBytes},
                                    b.getI8Type());
    Value packed = b.create<memref::AllocOp>(loc, packedTy);
    auto dynamic = ShapedType::kDynamic;
    auto inputTy = MemRefType::get(
        {dynamic, dynamic}, b.getF32Type(),
        StridedLayoutAttr::get(b.getContext(), dynamic, {dynamic, dynamic}));
    auto bytesTy = MemRefType::get({dynamic}, b.getI8Type());
    auto module = op->getParentOfType<ModuleOp>();
    if (!module.lookupSymbol<func::FuncOp>("mxfp8_quant")) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto fn = b.create<func::FuncOp>(
          loc, "mxfp8_quant",
          b.getFunctionType({inputTy, bytesTy, b.getI64Type(), b.getI64Type()},
                            {}));
      fn.setPrivate();
      fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    }
    b.create<func::CallOp>(
        loc, "mxfp8_quant", TypeRange{},
        ValueRange{b.create<memref::CastOp>(loc, inputTy, op.getInput()),
                   b.create<memref::CastOp>(loc, bytesTy, packed),
                   createI64Const(b, loc, rows),
                   createI64Const(b, loc, chunk)});
    Value limit = b.create<arith::ConstantIndexOp>(loc, N),
          step = b.create<arith::ConstantIndexOp>(loc, 16);
    auto panels = b.create<scf::ForOp>(loc, zero, limit, step);
    b.setInsertionPointToStart(panels.getBody());
    Value n = panels.getInductionVar();
    Value aBank = allocBank(b, loc, 1, 1), wBank = allocBank(b, loc, 1, 1),
          cBank = allocBank(b, loc, 1, 1);
    auto outPackTy = MemRefType::get({rows * 4, 4}, b.getF32Type());
    Value outPack = b.create<memref::AllocOp>(loc, outPackTy);
    for (int64_t m = 0; m < M; m += rows) {
      Value state = cBank;
      for (int64_t part = 0; part < chunks; ++part) {
        Value aOffset = b.create<arith::ConstantIndexOp>(
            loc, (m / rows * chunks + part) * aBytes);
        Value panel = b.create<arith::DivUIOp>(loc, n, step);
        Value wOffset = b.create<arith::MulIOp>(
            loc,
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, panel, b.create<arith::ConstantIndexOp>(loc, chunks)),
                b.create<arith::ConstantIndexOp>(loc, part)),
            b.create<arith::ConstantIndexOp>(loc, bBytes));
        auto bankView = [&](Value source, Value offset, int64_t size) -> Value {
          Value slice = b.create<memref::SubViewOp>(
              loc, source, SmallVector<OpFoldResult>{offset},
              SmallVector<OpFoldResult>{b.getIndexAttr(size)},
              SmallVector<OpFoldResult>{b.getIndexAttr(1)});
          auto ty = MemRefType::get(
              {size / 16, 16}, b.getI8Type(),
              StridedLayoutAttr::get(b.getContext(), dynamic, {16, 1}));
          return b.create<memref::ExpandShapeOp>(
              loc, ty, slice, ArrayRef<ReassociationIndices>{{0, 1}});
        };
        aBank = mvinBank(b, loc, bankView(packed, aOffset, aBytes), aBank,
                         aBytes / 16);
        wBank = mvinBank(b, loc, bankView(op.getWeight(), wOffset, bBytes),
                         wBank, bBytes / 16);
        state = b.create<BankMXFP8Op>(
            loc, b.getI64Type(), aBank, wBank, state,
            createI64Const(b, loc, matrixRs2(rows, 16, chunk)),
            createI1Const(b, loc, part == 0),
            createI1Const(b, loc, part == chunks - 1),
            createI64Const(b, loc, 0));
      }
      cBank = mvoutBank(b, loc, outPack, state, rows * 4);
      Value end = b.create<arith::MinUIOp>(
          loc, step, b.create<arith::SubIOp>(loc, limit, n));
      auto rr = b.create<scf::ForOp>(
          loc, zero,
          b.create<arith::ConstantIndexOp>(loc, std::min(rows, M - m)), one);
      b.setInsertionPointToStart(rr.getBody());
      auto cc = b.create<scf::ForOp>(loc, zero, end, one);
      b.setInsertionPointToStart(cc.getBody());
      Value r = rr.getInductionVar(), c = cc.getInductionVar(),
            four = b.create<arith::ConstantIndexOp>(loc, 4);
      Value pr =
          b.create<arith::AddIOp>(loc, b.create<arith::MulIOp>(loc, r, four),
                                  b.create<arith::DivUIOp>(loc, c, four));
      Value pc = b.create<arith::RemUIOp>(loc, c, four);
      Value v = b.create<memref::LoadOp>(loc, outPack, ValueRange{pr, pc});
      b.create<memref::StoreOp>(
          loc, v, op.getOutput(),
          ValueRange{b.create<arith::AddIOp>(
                         loc, r, b.create<arith::ConstantIndexOp>(loc, m)),
                     b.create<arith::AddIOp>(loc, n, c)});
      b.setInsertionPointAfter(rr);
    }
    releaseBank(b, loc, aBank);
    releaseBank(b, loc, wBank);
    releaseBank(b, loc, cBank);
    b.create<memref::DeallocOp>(loc, outPack);
    b.setInsertionPointAfter(panels);
    b.create<memref::DeallocOp>(loc, packed);
    b.eraseOp(op);
    return success();
  }
};
