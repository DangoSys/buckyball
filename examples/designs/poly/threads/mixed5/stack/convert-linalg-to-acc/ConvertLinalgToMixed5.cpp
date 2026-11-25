#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mxint5gemvOps.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace buddy::acc;
using namespace buddy::frontend;

namespace {
class ConvertLinalgToMixed5Pass
    : public PassWrapper<ConvertLinalgToMixed5Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToMixed5Pass)
  StringRef getArgument() const final { return "convert-linalg-to-mixed5"; }
  StringRef getDescription() const final {
    return "Fuse MXINT8 by scaled INT5 products with their existing "
           "quantization";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<AccDialect, arith::ArithDialect, scf::SCFDialect,
             bufferization::BufferizationDialect, memref::MemRefDialect,
             tensor::TensorDialect,
             buddy::mxint5gemv_ball::Mxint5gemvBallDialect>();
  }
  void runOnOperation() override {
    SmallVector<FP32ToMXINT8Op> candidates;
    getOperation().walk([&](FP32ToMXINT8Op q) { candidates.push_back(q); });
    for (auto q : candidates) {
      auto function = q->getParentOfType<func::FuncOp>();
      auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
      if (profile && profile.getValue() != target::getTarget().hart)
        continue;
      auto mm = q.getInput().getDefiningOp<linalg::MatmulOp>();
      auto shape = q.getInput().getType();
      if (!mm || !mm.getResult(0).hasOneUse() || shape.getRank() != 2 ||
          shape.getDimSize(0) == 0 || shape.getDimSize(1) <= 0 ||
          shape.getDimSize(1) % 32 || q.getAxis() != 1)
        continue;
      SmallVector<bufferization::MaterializeInDestinationOp> destinations;
      SmallVector<tensor::ExtractSliceOp> planes;
      if (q.getResult().hasOneUse()) {
        auto dest = dyn_cast<bufferization::MaterializeInDestinationOp>(
            *q.getResult().getUsers().begin());
        if (!dest)
          continue;
        destinations.push_back(dest);
      } else {
        destinations.resize(2);
        planes.resize(2);
        for (Operation *user : q.getResult().getUsers()) {
          auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
          if (!slice || !slice.getResult().hasOneUse() ||
              slice.getStaticOffsets().size() != 2 ||
              slice.getStaticOffsets()[0] != 0 ||
              slice.getStaticStrides() != ArrayRef<int64_t>{1, 1})
            continue;
          int64_t n = shape.getDimSize(1);
          int plane = slice.getStaticOffsets()[1] == 0 ? 0 : 1;
          if (slice.getStaticOffsets()[1] != (plane ? n : 0) ||
              slice.getType().getShape() !=
                  ArrayRef<int64_t>{shape.getDimSize(0), plane ? n / 32 : n})
            continue;
          auto dest = dyn_cast<bufferization::MaterializeInDestinationOp>(
              *slice.getResult().getUsers().begin());
          if (!dest || destinations[plane])
            continue;
          destinations[plane] = dest;
          planes[plane] = slice;
        }
        if (!destinations[0] || !destinations[1] ||
            std::distance(q.getResult().use_begin(), q.getResult().use_end()) !=
                2)
          continue;
      }
      auto a = mm.getDpsInputs()[0].getDefiningOp<MXINT8ToFP32Op>();
      Value w = mm.getDpsInputs()[1];
      auto collapse = w.getDefiningOp<tensor::CollapseShapeOp>();
      auto transpose =
          collapse ? collapse.getSrc().getDefiningOp<linalg::TransposeOp>()
                   : linalg::TransposeOp();
      if (!a || !transpose ||
          transpose.getPermutation() != ArrayRef<int64_t>{1, 0, 2})
        continue;
      auto weight = transpose.getInput().getDefiningOp<MXINT5ToFP32Op>();
      if (!weight)
        continue;
      auto at = a.getResult().getType();
      auto wt = weight.getResult().getType();
      int64_t rows = shape.getDimSize(0), n = shape.getDimSize(1),
              k = at.getDimSize(1);
      if (at.getRank() != 2 || at.getDimSize(0) != rows || k <= 0 ||
          k > 32768 || k % 512 || a.getAxis() != 1 || wt.getRank() != 3 ||
          wt.getDimSize(0) <= 0 || wt.getDimSize(1) != n ||
          wt.getDimSize(2) <= 0 || wt.getDimSize(2) % 512 ||
          wt.getDimSize(0) * wt.getDimSize(2) != k || weight.getAxis() != 2)
        continue;
      auto maps = mm.getIndexingMapsArray();
      auto ctx = &getContext();
      auto m = getAffineDimExpr(0, ctx), c = getAffineDimExpr(1, ctx),
           d = getAffineDimExpr(2, ctx);
      if (maps != SmallVector<AffineMap>{AffineMap::get(3, 0, {m, d}, ctx),
                                         AffineMap::get(3, 0, {c, d}, ctx),
                                         AffineMap::get(3, 0, {m, c}, ctx)})
        continue;
      auto init = mm.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
      if (!init || !matchPattern(init.getInputs()[0], m_AnyZeroFloat()))
        continue;
      if (q.getResult().getType().getShape() !=
          ArrayRef<int64_t>{rows, n + n / 32}) {
        q.emitError("GEMV needs codes followed by one exponent per 32 outputs");
        return signalPassFailure();
      }
      // The ABI proves contiguous bytes within a row. Outer rows may be
      // views of a larger packed tensor and keep their actual strides.
      auto rowLayout = [&](Value value) {
        SmallVector<int64_t> strides;
        auto arg = dyn_cast<BlockArgument>(value);
        if (!arg || arg.getOwner() != &function.getBody().front())
          return strides;
        auto type = cast<RankedTensorType>(value.getType());
        Attribute layout = function.getArgAttr(arg.getArgNumber(),
                                               "bufferization.buffer_layout");
        MemRefType memory;
        if (auto map = dyn_cast_or_null<AffineMapAttr>(layout))
          memory = MemRefType::get(type.getShape(), type.getElementType(),
                                   map.getValue());
        else if (auto map = dyn_cast_or_null<StridedLayoutAttr>(layout))
          memory = MemRefType::get(type.getShape(), type.getElementType(), map);
        else
          return strides;
        return memory.getStridesAndOffset().first;
      };
      auto as = rowLayout(a.getInput()), ws = rowLayout(weight.getInput());
      bool outputsLegal = true;
      for (auto [part, dest] : llvm::enumerate(destinations)) {
        auto stride = rowLayout(dest.getDest());
        int64_t width =
            destinations.size() == 1 ? n + n / 32 : (part ? n / 32 : n);
        outputsLegal &=
            stride.size() == 2 && stride[1] == 1 && stride[0] >= width;
      }
      int64_t packedWeight = wt.getDimSize(2) * 5 / 8 + wt.getDimSize(2) / 16;
      if (as.size() != 2 || as[1] != 1 || as[0] < k + k / 32 ||
          ws.size() != 3 || ws[2] != 1 || ws[1] != packedWeight ||
          ws[0] < n * packedWeight || !outputsLegal)
        continue;
      const auto &target = target::getTarget();
      int64_t chunk = wt.getDimSize(2), packed = chunk * 5 / 8 + chunk / 16;
      auto packedA = a.getInput().getType();
      auto packedW = weight.getInput().getType();
      auto packedOut = q.getResult().getType();
      if (packedA.getShape() != ArrayRef<int64_t>{rows, k + k / 32} ||
          packedW.getShape() != ArrayRef<int64_t>{k / chunk, n, packed} ||
          packedOut.getShape() != ArrayRef<int64_t>{rows, n + n / 32} ||
          (rows == 1 ? k + k / 32 : chunk + chunk / 32) >
              target.bankDepth * 16 ||
          8 * packed > target.bankDepth * 16)
        continue;
      if (target.bankDepth != 1024 || target.bankWidthBits != 128 ||
          std::min(target.bankNum, target.privateBankMax) < 6) {
        q.emitError("GEMV needs six private 16 KiB banks");
        return signalPassFailure();
      }
      OpBuilder b(q);
      auto loc = q.getLoc();
      auto idx = [&](int64_t x) -> Value {
        return arith::ConstantIndexOp::create(b, loc, x);
      };
      auto buffer = [&](Value tensor) -> Value {
        auto t = cast<RankedTensorType>(tensor.getType());
        return bufferization::ToBufferOp::create(
            b, loc,
            MemRefType::get(
                t.getShape(), t.getElementType(),
                StridedLayoutAttr::get(
                    ctx, ShapedType::kDynamic,
                    SmallVector<int64_t>(t.getRank(), ShapedType::kDynamic))),
            tensor, true);
      };
      Value am = buffer(a.getInput()), wm = buffer(weight.getInput());
      SmallVector<Value> outputs;
      SmallVector<memref::ExtractStridedMetadataOp> metadata;
      for (auto dest : destinations) {
        outputs.push_back(buffer(dest.getDest()));
        metadata.push_back(
            memref::ExtractStridedMetadataOp::create(b, loc, outputs.back()));
      }
      auto ai = memref::ExtractStridedMetadataOp::create(b, loc, am);
      auto wi = memref::ExtractStridedMetadataOp::create(b, loc, wm);
      auto add = [&](Value x, Value y) -> Value {
        return arith::AddIOp::create(b, loc, x, y);
      };
      auto mul = [&](Value x, int64_t y) -> Value {
        return arith::MulIOp::create(b, loc, x, idx(y));
      };
      auto range = [&](Value mem, Value offset, int64_t bytes) -> Value {
        return memref::ReinterpretCastOp::create(
            b, loc,
            MemRefType::get(
                {bytes}, b.getI8Type(),
                StridedLayoutAttr::get(ctx, ShapedType::kDynamic, {1})),
            mem, offset, ArrayRef<OpFoldResult>{b.getIndexAttr(bytes)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      auto writeOutput = [&](Value bank, Value state, Value row,
                             Value columnTile) -> Value {
        for (int part = 0; part < 2; ++part) {
          int destination = destinations.size() == 1 ? 0 : part;
          auto oi = metadata[destination];
          Value offset = part ? columnTile : mul(columnTile, 32);
          if (part && destinations.size() == 1)
            offset = add(offset, idx(n));
          Value start =
              add(oi.getOffset(),
                  add(arith::MulIOp::create(b, loc, row, oi.getStrides()[0]),
                      offset));
          Value mem = range(outputs[destination], start, part ? 1 : 32);
          bank = BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), mem, bank, add(state, idx(part ? 2 : 0)),
              idx(part ? 1 : 2), idx(16), 0, 1, idx(part ? 1 : 16), idx(0));
        }
        return bank;
      };
      Value ab = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value wb = BankAllocOp::create(b, loc, b.getI64Type(), 1, 4);
      Value cb = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      int64_t blocks = k / chunk;
      if (rows != 1) {
        int64_t tileRows = target.bankDepth / 17;
        // B stays resident across the row tile; the instruction template
        // loops over N and middle K chunks instead of growing with N*K.
        auto tile = scf::ForOp::create(b, loc, idx(0), ai.getSizes()[0],
                                       idx(tileRows), ValueRange{ab, wb, cb});
        tile->setAttr("hart.record", b.getUnitAttr());
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(tile.getBody());
          Value tileRow = tile.getInductionVar();
          Value count = arith::MinUIOp::create(
              b, loc, idx(tileRows),
              arith::SubIOp::create(b, loc, ai.getSizes()[0], tileRow));
          auto columns = scf::ForOp::create(b, loc, idx(0), idx(n / 32), idx(1),
                                            tile.getRegionIterArgs());
          columns->setAttr("hart.stream", b.getUnitAttr());
          {
            OpBuilder::InsertionGuard columnGuard(b);
            b.setInsertionPointToStart(columns.getBody());
            Value column = mul(columns.getInductionVar(), 32);
            Value ta = columns.getRegionIterArgs()[0],
                  tw = columns.getRegionIterArgs()[1],
                  tc = columns.getRegionIterArgs()[2];
            auto emitChunk = [&](Value block, bool accumulate, bool finalize) {
              for (int64_t group = 0; group < 4; ++group) {
                Value start = add(wi.getOffset(),
                                  add(arith::MulIOp::create(b, loc, block,
                                                            wi.getStrides()[0]),
                                      arith::MulIOp::create(
                                          b, loc, add(column, idx(group * 8)),
                                          wi.getStrides()[1])));
                tw = BankMvinStridedOp::create(
                    b, loc, b.getI64Type(), range(wm, start, 8 * packed), tw,
                    idx(0), idx(8 * packed / 16), idx(16), group, 1, idx(16),
                    idx(0));
              }
              auto row = scf::ForOp::create(b, loc, idx(0), count, idx(1),
                                            ValueRange{ta, tc});
              row->setAttr("hart.stream", b.getUnitAttr());
              {
                OpBuilder::InsertionGuard rowGuard(b);
                b.setInsertionPointToStart(row.getBody());
                Value r = row.getInductionVar(), global = add(tileRow, r);
                Value ra = row.getRegionIterArgs()[0],
                      rc = row.getRegionIterArgs()[1];
                Value inputRow = add(
                    ai.getOffset(),
                    arith::MulIOp::create(b, loc, global, ai.getStrides()[0]));
                for (int part = 0; part < 2; ++part) {
                  Value offset = part ? add(idx(k), mul(block, chunk / 32))
                                      : mul(block, chunk);
                  int64_t bytes = part ? chunk / 32 : chunk;
                  ra = BankMvinStridedOp::create(
                      b, loc, b.getI64Type(),
                      range(am, add(inputRow, offset), bytes), ra,
                      idx(part ? chunk / 16 : 0), idx(bytes / 16), idx(16), 0,
                      1, idx(16), idx(0));
                }
                Value state = mul(r, 17);
                Value cbase =
                    arith::IndexCastOp::create(b, loc, b.getI64Type(), state);
                rc = buddy::mxint5gemv_ball::BankGemvMxint5Op::create(
                    b, loc, b.getI64Type(), ra, tw, rc, chunk, 0, chunk / 16, 0,
                    cbase, accumulate, finalize, true);
                if (finalize)
                  rc =
                      writeOutput(rc, state, global, columns.getInductionVar());
                scf::YieldOp::create(b, loc, ValueRange{ra, rc});
              }
              ta = row.getResult(0);
              tc = row.getResult(1);
            };
            emitChunk(idx(0), false, blocks == 1);
            if (blocks > 2) {
              auto middle = scf::ForOp::create(b, loc, idx(0), idx(blocks - 2),
                                               idx(1), ValueRange{ta, tw, tc});
              middle->setAttr("hart.stream", b.getUnitAttr());
              {
                OpBuilder::InsertionGuard middleGuard(b);
                b.setInsertionPointToStart(middle.getBody());
                ta = middle.getRegionIterArgs()[0];
                tw = middle.getRegionIterArgs()[1];
                tc = middle.getRegionIterArgs()[2];
                emitChunk(add(middle.getInductionVar(), idx(1)), true, false);
                scf::YieldOp::create(b, loc, ValueRange{ta, tw, tc});
              }
              ta = middle.getResult(0);
              tw = middle.getResult(1);
              tc = middle.getResult(2);
            }
            if (blocks > 1)
              emitChunk(idx(blocks - 1), true, true);
            scf::YieldOp::create(b, loc, ValueRange{ta, tw, tc});
          }
          scf::YieldOp::create(b, loc, columns.getResults());
        }
        ab = tile.getResult(0);
        wb = tile.getResult(1);
        cb = tile.getResult(2);
      } else {
        // Static M1 keeps A resident across all N tiles. K bases are ISA
        // attributes, so only N is looped on this resident-A path.
        for (int part = 0; part < 2; ++part) {
          int64_t offset = part ? k : 0, rows = part ? k / 512 : k / 16;
          Value mem = range(am, add(ai.getOffset(), idx(offset)), rows * 16);
          ab = BankMvinStridedOp::create(b, loc, b.getI64Type(), mem, ab,
                                         idx(part ? k / 16 : 0), idx(rows),
                                         idx(16), 0, 1, idx(16), idx(0));
        }
        auto columns = scf::ForOp::create(b, loc, idx(0), idx(n / 32), idx(1),
                                          ValueRange{wb, cb});
        columns->setAttr("hart.stream", b.getUnitAttr());
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(columns.getBody());
          Value column = mul(columns.getInductionVar(), 32);
          Value tw = columns.getRegionIterArgs()[0],
                tc = columns.getRegionIterArgs()[1];
          for (int64_t block = 0; block < blocks; ++block) {
            for (int64_t group = 0; group < 4; ++group) {
              Value start = add(
                  wi.getOffset(),
                  add(mul(wi.getStrides()[0], block),
                      arith::MulIOp::create(b, loc, add(column, idx(group * 8)),
                                            wi.getStrides()[1])));
              tw = BankMvinStridedOp::create(
                  b, loc, b.getI64Type(), range(wm, start, 8 * packed), tw,
                  idx(0), idx(8 * packed / 16), idx(16), group, 1, idx(16),
                  idx(0));
            }
            tc = buddy::mxint5gemv_ball::BankGemvMxint5Op::create(
                b, loc, b.getI64Type(), ab, tw, tc, chunk, block * chunk / 16,
                k / 16 + block * chunk / 512, 0,
                arith::ConstantIntOp::create(b, loc, 0, 64), block != 0,
                block + 1 == blocks, false);
          }
          tc = writeOutput(tc, idx(0), idx(0), columns.getInductionVar());
          scf::YieldOp::create(b, loc, ValueRange{tw, tc});
        }
        wb = columns.getResult(0);
        cb = columns.getResult(1);
      }
      BankReleaseOp::create(b, loc, ab);
      BankReleaseOp::create(b, loc, wb);
      BankReleaseOp::create(b, loc, cb);
      for (auto dest : destinations) {
        dest.getResult().replaceAllUsesWith(dest.getDest());
        dest.erase();
      }
      for (auto plane : planes)
        plane.erase();
      q.erase();
      mm.erase();
    }
    RewritePatternSet patterns(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
    auto unsupported = getOperation().walk([&](MXINT5ToFP32Op op) {
      auto function = op->getParentOfType<func::FuncOp>();
      auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
      if (!profile || profile.getValue() != target::getTarget().hart)
        return WalkResult::advance();
      if (op->use_empty())
        return WalkResult::advance();
      op.emitError(
          "INT5 matrix needs a supported dense matmul-to-MXINT8 boundary");
      return WalkResult::interrupt();
    });
    if (unsupported.wasInterrupted())
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToMixed5Pass> registration;
} // namespace
