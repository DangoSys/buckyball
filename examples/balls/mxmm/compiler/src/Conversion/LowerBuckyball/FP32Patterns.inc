class FP32ToBanks : public OpRewritePattern<FP32MemMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FP32MemMatmulOp op,
                                PatternRewriter &b) const override {
    auto a = cast<MemRefType>(op.getLhs().getType()),
         w = cast<MemRefType>(op.getRhs().getType()),
         out = cast<MemRefType>(op.getOutput().getType());
    int rank = a.getRank();
    if ((rank != 2 && rank != 3) || w.getRank() != rank ||
        out.getRank() != rank || !a.hasStaticShape() || !w.hasStaticShape() ||
        !out.hasStaticShape() || !a.getElementType().isF32() ||
        !w.getElementType().isF32() || !out.getElementType().isF32())
      return op.emitError(
          "FP32 matmul requires static rank-2 or rank-3 FP32 operands");
    int64_t M = a.getShape()[rank - 2], K = a.getShape()[rank - 1],
            N = w.getShape()[rank - 1],
            batches = rank == 3 ? a.getShape()[0] : 1;
    if (M <= 0 || N <= 0 || K <= 0 || K % 4 || w.getShape()[rank - 2] != K ||
        out.getShape()[rank - 2] != M || out.getShape()[rank - 1] != N ||
        (rank == 3 &&
         (w.getShape()[0] != batches || out.getShape()[0] != batches)))
      return op.emitError(
          "FP32 matmul shape mismatch or K not divisible by four");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128)
      return op.emitError("FP32 matmul requires 128-bit bank rows");
    int64_t bytes = target.bankDepth * 16, rows = 0, columns = 0, chunk = 0;
    std::tuple<int64_t, int64_t, int64_t, int64_t> best{INT64_MAX, 0, 0, 0};
    for (int64_t m = M == 1 ? 1 : 16; m <= (M == 1 ? 1 : (M + 15) / 16 * 16);
         m += 16)
      for (int64_t n = 16; n <= (N + 15) / 16 * 16; n += 16) {
        if (m * n * 4 > bytes)
          continue;
        int64_t k = std::min(K, bytes / (4 * std::max(m, n))) / 4 * 4;
        if (!k)
          continue;
        auto score = std::tuple{((M + m - 1) / m) * ((N + n - 1) / n) *
                                    ((K + k - 1) / k),
                                -k, -m, -n};
        if (score < best) {
          best = score;
          rows = m;
          columns = n;
          chunk = k;
        }
      }
    if (!rows)
      return op.emitError("FP32 matmul cannot fit a bank block");
    SmallVector<int64_t> rhsStrides;
    int64_t rhsOffset;
    bool bankPack =
        target.bankNum >= 5 && target.rvvEnabled &&
        succeeded(w.getStridesAndOffset(rhsStrides, rhsOffset)) &&
        rhsStrides[rank - 2] == N && rhsStrides.back() == 1 &&
        K * N * 4 <= bytes && chunk == K && rows * columns * 4 >= 64;
    Location loc = op.getLoc();
    auto idx = [&](int64_t v) -> Value {
      return b.create<arith::ConstantIndexOp>(loc, v);
    };
    Value zero = idx(0), one = idx(1), four = idx(4);
    Value ab = allocBank(b, loc, 1, 1), wb = allocBank(b, loc, 1, 1),
          cb = allocBank(b, loc, 1, 1);
    auto packType = MemRefType::get({bytes / 16, 4}, b.getF32Type());
    Value ap = b.create<memref::AllocOp>(loc, packType),
          bp = b.create<memref::AllocOp>(loc, packType),
          cp = b.create<memref::AllocOp>(loc, packType);
    for (Value buffer : {ap, bp, cp})
      buffer.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(bytes));
    SmallVector<Value> banks{ab, wb, cb};
    if (bankPack) {
      Value input = allocBank(b, loc, 1, 1);
      Value descriptor = allocBank(b, loc, 1, 1);
      Value read = b.create<BankTransferOp>(loc, b.getI64Type(), ab, input);
      Value write = b.create<BankTransferOp>(loc, b.getI64Type(), wb, descriptor);
      write = b.create<BankTransferOp>(loc, b.getI64Type(), cb, write);
      auto loaded = b.create<BankKernelOp>(
          loc, b.getI64Type(), b.getI64Type(), read, write,
          FlatSymbolRefAttr::get(b.getContext(), "rvv_packmm"), ValueRange{
          createI64Const(b, loc, 0), createI64Const(b, loc, K),
          createI64Const(b, loc, N), createI64Const(b, loc, 0),
          createI64Const(b, loc, 0), createI64Const(b, loc, K),
          createI64Const(b, loc, columns), createI64Const(b, loc, rows),
          createI64Const(b, loc, 1), createI64Const(b, loc, 1),
          createI64Const(b, loc, op.getFusedAttr().getValue())});
      banks = {loaded.getReadOut(), loaded.getWriteOut()};
    }
    auto batchLoop = b.create<scf::ForOp>(loc, zero, idx(batches), one, banks);
    b.setInsertionPointToStart(batchLoop.getBody());
    Value batch = batchLoop.getInductionVar();
    SmallVector<Value> batchBanks(batchLoop.getRegionIterArgs());
    if (bankPack) {
      SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> sizes;
      for (int64_t dim : w.getShape())
        sizes.push_back(b.getIndexAttr(dim));
      SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
      if (rank == 3) {
        offsets[0] = batch;
        sizes[0] = b.getIndexAttr(1);
      }
      auto rowType = memref::SubViewOp::inferRankReducedResultType(
          {K, N}, w, offsets, sizes, strides);
      Value source = b.create<memref::SubViewOp>(loc, rowType, op.getRhs(),
                                                 offsets, sizes, strides);
      Value flat = b.create<memref::CollapseShapeOp>(
          loc, source, ArrayRef<ReassociationIndices>{{0, 1}});
      Value packed = b.create<memref::ExpandShapeOp>(
          loc, ArrayRef<int64_t>{K * N / 4, 4}, flat,
          ArrayRef<ReassociationIndices>{{0, 1}});
      batchBanks[0] = b.create<BankMvinOp>(
          loc, b.getI64Type(), packed, batchBanks[0],
          createI64Const(b, loc, K * N / 4), createI64Const(b, loc, 1),
          b.getI64IntegerAttr(0));
    }
    auto rowLoop =
        b.create<scf::ForOp>(loc, zero, idx(M), idx(rows), batchBanks);
    b.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    auto colLoop = b.create<scf::ForOp>(loc, zero, idx(N), idx(columns),
                                        rowLoop.getRegionIterArgs());
    b.setInsertionPointToStart(colLoop.getBody());
    Value column = colLoop.getInductionVar();
    ab = colLoop.getRegionIterArgs()[0];
    wb = colLoop.getRegionIterArgs()[1];
    cb = bankPack ? wb : colLoop.getRegionIterArgs()[2];
    for (int64_t begin = 0; begin < K; begin += chunk) {
      int64_t count = std::min(chunk, K - begin);
      for (int operand = 0; operand < (bankPack ? 1 : 2); ++operand) {
        Value source = operand ? op.getRhs() : op.getLhs();
        Value packed = operand ? bp : ap;
        auto sourceType = cast<MemRefType>(source.getType());
        int axis = operand ? rank - 2 : rank - 1;
        SmallVector<int64_t> strides;
        int64_t offset;
        if (failed(sourceType.getStridesAndOffset(strides, offset)))
          return op.emitError("FP32 matmul requires strided operands");
        int64_t blockRows = operand ? columns : rows;
        int64_t logicalRows = operand ? N : M;
        Value base = operand ? column : row;
        Value validRows = b.create<arith::MinUIOp>(
            loc, idx(blockRows),
            b.create<arith::SubIOp>(loc, idx(logicalRows), base));
        auto flatType = MemRefType::get({bytes / 4}, b.getF32Type());
        Value flatBuffer = b.create<memref::CollapseShapeOp>(
            loc, flatType, packed, ArrayRef<ReassociationIndices>{{0, 1}});
        auto rLoop = b.create<scf::ForOp>(loc, zero, validRows, one);
        b.setInsertionPointToStart(rLoop.getBody());
        Value r = rLoop.getInductionVar();
        Value global = b.create<arith::AddIOp>(loc, base, r);
        Value flat = b.create<arith::MulIOp>(loc, r, idx(count));
        if (strides[axis] == 1) {
          SmallVector<OpFoldResult> sourceOffsets(rank, b.getIndexAttr(0));
          SmallVector<OpFoldResult> sourceSizes(rank, b.getIndexAttr(1));
          SmallVector<OpFoldResult> sourceStrides(rank, b.getIndexAttr(1));
          if (rank == 3)
            sourceOffsets[0] = batch;
          sourceOffsets[rank - 2] = operand
                                        ? OpFoldResult(b.getIndexAttr(begin))
                                        : OpFoldResult(global);
          sourceOffsets[rank - 1] = operand
                                        ? OpFoldResult(global)
                                        : OpFoldResult(b.getIndexAttr(begin));
          sourceSizes[axis] = b.getIndexAttr(count);
          auto sourceRowType = memref::SubViewOp::inferRankReducedResultType(
              {count}, sourceType, sourceOffsets, sourceSizes, sourceStrides);
          Value sourceRow = b.create<memref::SubViewOp>(
              loc, sourceRowType, source, sourceOffsets, sourceSizes,
              sourceStrides);
          SmallVector<OpFoldResult> targetOffsets{flat};
          SmallVector<OpFoldResult> targetSizes{b.getIndexAttr(count)};
          SmallVector<OpFoldResult> targetStrides{b.getIndexAttr(1)};
          auto targetRowType = memref::SubViewOp::inferRankReducedResultType(
              {count}, flatType, targetOffsets, targetSizes, targetStrides);
          Value targetRow = b.create<memref::SubViewOp>(
              loc, targetRowType, flatBuffer, targetOffsets, targetSizes,
              targetStrides);
          b.create<memref::CopyOp>(loc, sourceRow, targetRow);
        } else {
          auto kLoop = b.create<scf::ForOp>(loc, zero, idx(count), one);
          b.setInsertionPointToStart(kLoop.getBody());
          Value k = kLoop.getInductionVar();
          Value kk = b.create<arith::AddIOp>(loc, k, idx(begin));
          SmallVector<Value> indices;
          if (rank == 3)
            indices.push_back(batch);
          indices.push_back(operand ? kk : global);
          indices.push_back(operand ? global : kk);
          Value value = b.create<memref::LoadOp>(loc, source, indices);
          Value destination = b.create<arith::AddIOp>(loc, flat, k);
          b.create<memref::StoreOp>(loc, value, flatBuffer,
                                    ValueRange{destination});
        }
        b.setInsertionPointAfter(rLoop);
        if (logicalRows % blockRows) {
          Value start = b.create<arith::MulIOp>(loc, validRows, idx(count * 4));
          Value pointer =
              b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, packed);
          Value address = b.create<arith::AddIOp>(loc, pointer, start);
          Value integer =
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), address);
          Value destination = b.create<LLVM::IntToPtrOp>(
              loc, LLVM::LLVMPointerType::get(b.getContext()), integer);
          Value length =
              b.create<arith::SubIOp>(loc, idx(blockRows * count * 4), start);
          Value byteLength =
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), length);
          Value byteZero = b.create<arith::ConstantIntOp>(loc, 0, 8);
          b.create<LLVM::MemsetOp>(loc, destination, byteZero, byteLength,
                                   false);
        }
      }
      if (bankPack) {
        ab = b.create<BankMvinOp>(
            loc, b.getI64Type(), ap, ab,
            createI64Const(b, loc, rows * count / 4),
            createI64Const(b, loc, 1), b.getI64IntegerAttr(1));
        auto packed = b.create<BankKernelOp>(
            loc, b.getI64Type(), b.getI64Type(), ab, wb,
            FlatSymbolRefAttr::get(b.getContext(), "rvv_packmm"), ValueRange{
            createI64Const(b, loc, 1), createI64Const(b, loc, K),
            createI64Const(b, loc, N), createI64Const(b, loc, begin),
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), column),
            createI64Const(b, loc, count), createI64Const(b, loc, columns),
            createI64Const(b, loc, rows), createI64Const(b, loc, begin == 0),
            createI64Const(b, loc, begin + count == K),
            createI64Const(b, loc, op.getFusedAttr().getValue())});
        ab = packed.getReadOut();
        wb = cb = packed.getWriteOut();
      } else {
        ab = mvinBank(b, loc, ap, ab, rows * count / 4);
        wb = mvinBank(b, loc, bp, wb, columns * count / 4);
        cb = b.create<BankFP32Op>(loc, b.getI64Type(), ab, wb, cb,
                                  createI64Const(b, loc,
                                                 packBits(rows, 0, 11) |
                                                     packBits(columns, 12, 23) |
                                                     packBits(count, 24, 35)),
                                  createI1Const(b, loc, begin == 0),
                                  createI1Const(b, loc, begin + count == K),
                                  createI64Const(b, loc, 0), op.getFusedAttr())
                 .getWrBankOut();
      }
    }
    if (bankPack)
      cb = b.create<BankMvoutOp>(
          loc, b.getI64Type(), cp, cb,
          createI64Const(b, loc, rows * columns / 4),
          createI64Const(b, loc, 1), b.getI64IntegerAttr(2));
    else
      cb = mvoutBank(b, loc, cp, cb, rows * columns / 4);
    b.create<FenceOp>(loc);
    SmallVector<int64_t> outputStrides;
    int64_t outputOffset;
    if (failed(out.getStridesAndOffset(outputStrides, outputOffset)))
      return op.emitError("FP32 matmul requires a strided output");
    Value validColumns = b.create<arith::MinUIOp>(
        loc, idx(columns), b.create<arith::SubIOp>(loc, idx(N), column));
    auto rr = b.create<scf::ForOp>(
        loc, zero,
        b.create<arith::MinUIOp>(loc, idx(rows),
                                 b.create<arith::SubIOp>(loc, idx(M), row)),
        one);
    b.setInsertionPointToStart(rr.getBody());
    Value r = rr.getInductionVar();
    if (outputStrides.back() == 1) {
      auto flatType = MemRefType::get({bytes / 4}, b.getF32Type());
      Value flat = b.create<memref::CollapseShapeOp>(
          loc, flatType, cp, ArrayRef<ReassociationIndices>{{0, 1}});
      SmallVector<OpFoldResult> offsets{
          b.create<arith::MulIOp>(loc, r, idx(columns)).getResult()};
      SmallVector<OpFoldResult> sizes{validColumns}, strides{b.getIndexAttr(1)};
      auto sourceType = memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic}, flatType, offsets, sizes, strides);
      Value source = b.create<memref::SubViewOp>(loc, sourceType, flat, offsets,
                                                 sizes, strides);
      SmallVector<OpFoldResult> outOffsets(rank, b.getIndexAttr(0));
      SmallVector<OpFoldResult> outSizes(rank, b.getIndexAttr(1));
      SmallVector<OpFoldResult> outSteps(rank, b.getIndexAttr(1));
      if (rank == 3)
        outOffsets[0] = batch;
      outOffsets[rank - 2] = b.create<arith::AddIOp>(loc, row, r).getResult();
      outOffsets[rank - 1] = column;
      outSizes.back() = validColumns;
      auto rowType = memref::SubViewOp::inferRankReducedResultType(
          {ShapedType::kDynamic}, out, outOffsets, outSizes, outSteps);
      Value destination = b.create<memref::SubViewOp>(
          loc, rowType, op.getOutput(), outOffsets, outSizes, outSteps);
      b.create<memref::CopyOp>(loc, source, destination);
    } else {
      auto cc = b.create<scf::ForOp>(loc, zero, validColumns, one);
      b.setInsertionPointToStart(cc.getBody());
      Value c = cc.getInductionVar();
      Value flat = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, r, idx(columns)), c);
      Value result = b.create<memref::LoadOp>(
          loc, cp,
          ValueRange{b.create<arith::DivUIOp>(loc, flat, four),
                     b.create<arith::RemUIOp>(loc, flat, four)});
      SmallVector<Value> outputIndices;
      if (rank == 3)
        outputIndices.push_back(batch);
      outputIndices.push_back(b.create<arith::AddIOp>(loc, row, r));
      outputIndices.push_back(b.create<arith::AddIOp>(loc, column, c));
      b.create<memref::StoreOp>(loc, result, op.getOutput(), outputIndices);
    }
    b.setInsertionPointAfter(rr);
    SmallVector<Value> nextBanks{ab, wb, cb};
    if (bankPack)
      nextBanks = {ab, cb};
    b.create<scf::YieldOp>(loc, nextBanks);
    b.setInsertionPointAfter(colLoop);
    b.create<scf::YieldOp>(loc, colLoop.getResults());
    b.setInsertionPointAfter(rowLoop);
    b.create<scf::YieldOp>(loc, rowLoop.getResults());
    b.setInsertionPointAfter(batchLoop);
    if (bankPack) {
      auto finished = b.create<BankKernelOp>(
          loc, b.getI64Type(), b.getI64Type(), batchLoop.getResult(0),
          batchLoop.getResult(1),
          FlatSymbolRefAttr::get(b.getContext(), "rvv_packmm"), ValueRange{
          createI64Const(b, loc, 2), createI64Const(b, loc, K),
          createI64Const(b, loc, N), createI64Const(b, loc, 0),
          createI64Const(b, loc, 0), createI64Const(b, loc, K),
          createI64Const(b, loc, columns), createI64Const(b, loc, rows),
          createI64Const(b, loc, 1), createI64Const(b, loc, 1),
          createI64Const(b, loc, op.getFusedAttr().getValue())});
      releaseBank(b, loc, finished.getReadOut());
      releaseBank(b, loc, finished.getWriteOut());
    } else {
      for (Value bank : batchLoop.getResults())
        releaseBank(b, loc, bank);
    }
    for (Value buffer : {ap, bp, cp})
      b.create<memref::DeallocOp>(loc, buffer);
    b.eraseOp(op);
    return success();
  }
};
