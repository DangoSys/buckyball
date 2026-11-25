#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "transposeOps.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult lowerGobanTransposeOutputs(ModuleOp module) {
  SmallVector<linalg::TransposeOp> transposes;
  module.walk([&](linalg::TransposeOp op) {
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    if (type.getElementType().isF32() && op->getResult(0).hasOneUse() &&
        isa<bufferization::MaterializeInDestinationOp>(
            *op->getResult(0).getUsers().begin()))
      transposes.push_back(op);
  });
  for (auto op : transposes) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = cast<RankedTensorType>(op.getInput().getType());
    int64_t rank = type.getRank();
    const auto &target = target::getTarget();
    auto destination = cast<bufferization::MaterializeInDestinationOp>(
        *op->getResult(0).getUsers().begin());
    Value input = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(type.getShape(), b.getF32Type()),
        op.getInput());
    auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
    Value output = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(outputType.getShape(), b.getF32Type()),
        destination.getDest());
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto attr = [&](int64_t value) -> OpFoldResult {
      return b.getIndexAttr(value);
    };
    if (rank > 2) {
      int64_t width = type.getShape().back();
      if (op.getPermutation().back() != rank - 1 || width % 4 ||
          width / 4 > target.bankDepth)
        return op.emitError("transpose requires contiguous whole-word blocks");
      Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      {
        OpBuilder::InsertionGuard guard(b);
        SmallVector<OpFoldResult> source(rank, attr(0)),
            destination(rank, attr(0));
        SmallVector<OpFoldResult> sizes(rank, attr(1)), strides(rank, attr(1));
        sizes.back() = attr(width);
        for (int64_t axis = 0; axis < rank - 1; ++axis) {
          auto loop = scf::ForOp::create(
              b, loc, index(0), index(outputType.getDimSize(axis)), index(1));
          loop->setAttr("hart.stream", b.getUnitAttr());
          b.setInsertionPointToStart(loop.getBody());
          destination[axis] = loop.getInductionVar();
          source[op.getPermutation()[axis]] = loop.getInductionVar();
        }
        Value from =
            memref::SubViewOp::create(b, loc, input, source, sizes, strides);
        Value to = memref::SubViewOp::create(b, loc, output, destination, sizes,
                                             strides);
        Value loaded = BankMvinOp::create(b, loc, b.getI64Type(), from, bank,
                                          width / 4, 1);
        BankMvoutOp::create(b, loc, b.getI64Type(), to, loaded, width / 4, 1);
      }
      BankReleaseOp::create(b, loc, bank);
      Value written = bufferization::ToTensorOp::create(b, loc, outputType,
                                                        output, true, true);
      destination.getResult().replaceAllUsesWith(written);
      destination.erase();
      op.erase();
      continue;
    }
    int64_t rows = type.getDimSize(0), columns = type.getDimSize(1);
    if (op.getPermutation() != ArrayRef<int64_t>{1, 0} || rows % 4 ||
        columns % 4 || rows > target.bankDepth || target.bankWidthBits != 128)
      return op.emitError("Goban FP32 transpose exceeds the bank geometry");
    // Both banks have G groups. Each input group gathers four columns;
    // transpose packs their output densely into G groups of the same depth.
    int64_t tileColumns = target.bankNum / 2 * 4;
    auto tiles = scf::ForOp::create(
        b, loc, index(0), index((columns + tileColumns - 1) / tileColumns),
        index(1));
    tiles->setAttr("hart.record", b.getUnitAttr());
    b.setInsertionPointToStart(tiles.getBody());
    Value first = arith::MulIOp::create(b, loc, tiles.getInductionVar(),
                                        index(tileColumns));
    auto emitTile = [&](int64_t count) {
      int64_t groups = count / 4;
      auto stream = scf::ForOp::create(b, loc, index(0), index(1), index(1));
      stream->setAttr("hart.stream", b.getUnitAttr());
      b.setInsertionPointToStart(stream.getBody());
      Value source = BankAllocOp::create(b, loc, b.getI64Type(), 1, groups);
      for (int64_t group = 0; group < groups; ++group) {
        Value slice = memref::SubViewOp::create(
            b, loc, input,
            ArrayRef<OpFoldResult>{
                attr(0), arith::AddIOp::create(b, loc, first, index(group * 4))
                             .getResult()},
            ArrayRef<OpFoldResult>{attr(rows), attr(4)},
            ArrayRef<OpFoldResult>{attr(1), attr(1)});
        source = BankMvinGroupOp::create(b, loc, b.getI64Type(), slice, source,
                                         group, rows, columns / 4);
      }
      Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, groups);
      result = ::buddy::transpose_ball::BankTransposeF32Op::create(
          b, loc, b.getI64Type(), source, result, rows, count);
      Value slice = memref::SubViewOp::create(
          b, loc, output, ArrayRef<OpFoldResult>{first, attr(0)},
          ArrayRef<OpFoldResult>{attr(count), attr(rows)},
          ArrayRef<OpFoldResult>{attr(1), attr(1)});
      BankMvoutOp::create(b, loc, b.getI64Type(), slice, result, rows, 1);
      BankReleaseOp::create(b, loc, source);
      BankReleaseOp::create(b, loc, result);
      b.setInsertionPointAfter(stream);
    };
    if (columns < tileColumns) {
      emitTile(columns);
    } else if (columns % tileColumns) {
      Value full = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                         tiles.getInductionVar(),
                                         index(columns / tileColumns));
      auto branch = scf::IfOp::create(b, loc, full, true);
      b.setInsertionPointToStart(branch.thenBlock());
      emitTile(tileColumns);
      b.setInsertionPointToStart(branch.elseBlock());
      emitTile(columns % tileColumns);
    } else {
      emitTile(tileColumns);
    }
    b.setInsertionPointAfter(tiles);
    Value written = bufferization::ToTensorOp::create(b, loc, outputType,
                                                      output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
