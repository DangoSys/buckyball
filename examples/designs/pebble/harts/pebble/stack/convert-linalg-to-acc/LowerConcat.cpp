#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::acc;

namespace pebble {
LogicalResult lowerConcatColumns(::buddy::frontend::ConcatColumnsOp op) {
  auto type = cast<RankedTensorType>(op.getResult().getType());
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      type.getDimSize(1) <= 0 || op.getInputs().empty())
    return op.emitOpError("requires static matrices");
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](int64_t x) -> Value {
    return arith::ConstantIndexOp::create(b, loc, x);
  };
  int64_t elementBytes = type.getElementTypeBitWidth() / 8;
  int64_t rowBytes = target::getTarget().bankWidthBits / 8;
  int64_t outputStride = type.getDimSize(1) * elementBytes;
  int64_t span = (target::getTarget().dmaMaxRanges - 1) * 4096;
  int64_t tileRows =
      outputStride <= 4095
          ? std::min<int64_t>(target::getTarget().bankDepth,
                              1 + (span - rowBytes) / outputStride)
          : 1;
  Value outputPitch = index(outputStride <= 4095 ? outputStride : rowBytes);
  Value capacity = index(tileRows);
  SmallVector<Value> inputs, pitches;
  SmallVector<int64_t> columns;
  int64_t totalColumns = 0;
  for (Value tensor : op.getInputs()) {
    auto source = cast<RankedTensorType>(tensor.getType());
    if (!source.hasStaticShape() || source.getRank() != 2 ||
        source.getDimSize(0) != type.getDimSize(0) ||
        source.getElementType() != type.getElementType())
      return op.emitOpError("requires matching matrix rows and element types");
    if (source.getDimSize(1) == 0)
      continue;
    auto layout = StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic,
                                         {ShapedType::kDynamic, 1});
    Value input = bufferization::ToBufferOp::create(
        b, loc,
        MemRefType::get(source.getShape(), source.getElementType(), layout),
        tensor, true);
    auto metadata = memref::ExtractStridedMetadataOp::create(b, loc, input);
    Value stride = arith::MulIOp::create(b, loc, metadata.getStrides()[0],
                                         index(elementBytes));
    auto [pitch, rows] =
        dmaRows(b, loc, stride, rowBytes, target::getTarget().bankDepth);
    capacity = arith::MinUIOp::create(b, loc, capacity, rows);
    inputs.push_back(input);
    pitches.push_back(pitch);
    columns.push_back(source.getDimSize(1));
    totalColumns += source.getDimSize(1);
  }
  if (totalColumns != type.getDimSize(1))
    return op.emitOpError("result columns must equal the input column sum");
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(type.getShape(), type.getElementType()));
  Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
  // Traverse output row tiles, respecting runtime source pitches within each
  // tile.
  int64_t tiles = (type.getDimSize(0) + tileRows - 1) / tileRows;
  auto outer = scf::ForOp::create(b, loc, index(0), index(tiles), index(1),
                                  ValueRange{bank});
  {
    OpBuilder::InsertionGuard outerGuard(b);
    b.setInsertionPointToStart(outer.getBody());
    Value first =
        arith::MulIOp::create(b, loc, outer.getInductionVar(), index(tileRows));
    Value end = arith::MinUIOp::create(
        b, loc, index(type.getDimSize(0)),
        arith::AddIOp::create(b, loc, first, index(tileRows)));
    auto loop = scf::ForOp::create(b, loc, first, end, capacity,
                                   outer.getRegionIterArgs());
    {
      OpBuilder::InsertionGuard loopGuard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value row = loop.getInductionVar();
      Value count = arith::MinUIOp::create(
          b, loc, capacity, arith::SubIOp::create(b, loc, end, row));
      Value moved = loop.getRegionIterArgs()[0];
      int64_t offset = 0;
      for (size_t input = 0; input < inputs.size(); ++input) {
        for (int64_t column = 0; column < columns[input];
             column += rowBytes / elementBytes) {
          int64_t bytes =
              std::min(rowBytes, (columns[input] - column) * elementBytes);
          Value memory = memref::SubViewOp::create(
              b, loc, inputs[input],
              ArrayRef<OpFoldResult>{row, b.getIndexAttr(column)},
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
          moved = BankMvinStridedOp::create(
              b, loc, b.getI64Type(), memory, moved, index(0), count,
              pitches[input], 0, 1, index(bytes), index(0));
          memory = memref::SubViewOp::create(
              b, loc, output,
              ArrayRef<OpFoldResult>{row, b.getIndexAttr(offset + column)},
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
          moved = BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), memory, moved, index(0), count,
              outputPitch, 0, 1, index(bytes), index(0));
        }
        offset += columns[input];
      }
      scf::YieldOp::create(b, loc, moved);
    }
    scf::YieldOp::create(b, loc, loop.getResults());
  }
  BankReleaseOp::create(b, loc, outer.getResult(0));
  Value result =
      bufferization::ToTensorOp::create(b, loc, type, output, true, true);
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}
} // namespace pebble
