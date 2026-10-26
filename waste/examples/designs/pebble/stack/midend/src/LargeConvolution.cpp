#include "Buckyball/BuckyballOps.h"
#include "Trace/TraceOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult lowerPebbleLargeConvolutions(func::FuncOp function) {
  SmallVector<linalg::GenericOp> convolutions;
  function.walk([&](linalg::GenericOp op) {
    if (op->hasAttr("buckyball.quantized_conv2d") &&
        op->getAttrOfType<IntegerAttr>("kernel").getInt() > 7)
      convolutions.push_back(op);
  });
  for (auto op : convolutions) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    Value input = op.getInputs()[0], weight = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    auto in = cast<MemRefType>(input.getType()).getShape();
    auto wt = cast<MemRefType>(weight.getType()).getShape();
    auto outType = cast<MemRefType>(output.getType());
    auto out = outType.getShape();
    int64_t kernel = op->getAttrOfType<IntegerAttr>("kernel").getInt();
    int64_t stride = op->getAttrOfType<IntegerAttr>("stride").getInt();
    int64_t padding = op->getAttrOfType<IntegerAttr>("pad_low").getInt();
    bool nchw = outType.getElementType().isF32() &&
                !op->getAttrOfType<BoolAttr>("output_nhwc").getValue();
    int64_t height = out[nchw ? 2 : 1], width = out[nchw ? 3 : 2];
    int64_t rows = height * width, columns = wt[0];
    int64_t reduction = in[3] * kernel * kernel;
    int64_t packedReduction = ((reduction + 15) / 16) * 16;
    if (in[0] != 1 || wt[1] != in[3] ||
        op->getAttrOfType<IntegerAttr>("activation").getInt() > 1)
      return op.emitError("large Pebble convolution requires batch one and "
                          "linear/ReLU activation");
    auto idx = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    Value zero = idx(0), one = idx(1);
    Value zeroByte = arith::ConstantIntOp::create(b, loc, 0, 8);
    Value patches = memref::AllocOp::create(
        b, loc, MemRefType::get({rows, packedReduction}, b.getI8Type()));
    Value weights = memref::AllocOp::create(
        b, loc, MemRefType::get({packedReduction, columns}, b.getI8Type()));
    Value result = memref::AllocOp::create(
        b, loc, MemRefType::get({rows, columns}, b.getF32Type()));
    linalg::FillOp::create(b, loc, zeroByte, patches);
    linalg::FillOp::create(b, loc, zeroByte, weights);
    auto trace = op->getAttrOfType<DictionaryAttr>("trace_node");
    if (trace && cast<BoolAttr>(trace.get("cycle")).getValue()) {
      auto start = ::buddy::trace::StartOp::create(
          b, loc, cast<IntegerAttr>(trace.get("id")), b.getBoolAttr(true),
          b.getBoolAttr(false), cast<StringAttr>(trace.get("tag")));
      start->setAttr("id_path", trace.get("id_path"));
    }
    auto rowLoop = scf::ForOp::create(b, loc, zero, idx(rows), one);
    b.setInsertionPointToStart(rowLoop.getBody());
    auto kLoop = scf::ForOp::create(b, loc, zero, idx(reduction), one);
    b.setInsertionPointToStart(kLoop.getBody());
    Value row = rowLoop.getInductionVar(), k = kLoop.getInductionVar();
    Value channel = arith::DivUIOp::create(b, loc, k, idx(kernel * kernel));
    Value ky = arith::RemUIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, k, idx(kernel)), idx(kernel));
    Value kx = arith::RemUIOp::create(b, loc, k, idx(kernel));
    Value y = arith::SubIOp::create(
        b, loc,
        arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(
                b, loc, arith::DivUIOp::create(b, loc, row, idx(width)),
                idx(stride)),
            ky),
        idx(padding));
    Value x = arith::SubIOp::create(
        b, loc,
        arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(
                b, loc, arith::RemUIOp::create(b, loc, row, idx(width)),
                idx(stride)),
            kx),
        idx(padding));
    Value valid = arith::AndIOp::create(
        b, loc,
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, y, idx(in[1])),
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, x,
                              idx(in[2])));
    auto present = scf::IfOp::create(b, loc, valid, false);
    b.setInsertionPointToStart(present.thenBlock());
    Value value =
        memref::LoadOp::create(b, loc, input, ValueRange{zero, y, x, channel});
    memref::StoreOp::create(b, loc, value, patches, ValueRange{row, k});
    b.setInsertionPointAfter(rowLoop);
    auto outputChannel = scf::ForOp::create(b, loc, zero, idx(columns), one);
    b.setInsertionPointToStart(outputChannel.getBody());
    auto weightK = scf::ForOp::create(b, loc, zero, idx(reduction), one);
    b.setInsertionPointToStart(weightK.getBody());
    k = weightK.getInductionVar();
    channel = arith::DivUIOp::create(b, loc, k, idx(kernel * kernel));
    ky = arith::RemUIOp::create(
        b, loc, arith::DivUIOp::create(b, loc, k, idx(kernel)), idx(kernel));
    kx = arith::RemUIOp::create(b, loc, k, idx(kernel));
    Value co = outputChannel.getInductionVar();
    value =
        memref::LoadOp::create(b, loc, weight, ValueRange{co, channel, ky, kx});
    memref::StoreOp::create(b, loc, value, weights, ValueRange{k, co});
    b.setInsertionPointAfter(outputChannel);
    auto region = MegaKernelOp::create(b, loc, patches, result);
    region.getBody().emplaceBlock();
    b.setInsertionPointToStart(&region.getBody().front());
    MegaMatmulOp::create(b, loc, patches, weights, op.getInputs()[2],
                         op.getInputs()[3], op.getInputs()[4], result,
                         op->getAttrOfType<IntegerAttr>("activation"),
                         op->getAttrOfType<FloatAttr>("output_scale"));
    MegaYieldOp::create(b, loc);
    b.setInsertionPointAfter(region);
    auto copyRow = scf::ForOp::create(b, loc, zero, idx(rows), one);
    b.setInsertionPointToStart(copyRow.getBody());
    auto copyChannel = scf::ForOp::create(b, loc, zero, idx(columns), one);
    b.setInsertionPointToStart(copyChannel.getBody());
    row = copyRow.getInductionVar();
    co = copyChannel.getInductionVar();
    value = memref::LoadOp::create(b, loc, result, ValueRange{row, co});
    if (outType.getElementType().isInteger(8)) {
      value = math::RoundEvenOp::create(b, loc, value);
      Value low = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(-128.0));
      Value high = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(127.0));
      value = arith::MaximumFOp::create(b, loc, value, low);
      value = arith::MinimumFOp::create(b, loc, value, high);
      value = arith::FPToSIOp::create(b, loc, b.getI8Type(), value);
    }
    y = arith::DivUIOp::create(b, loc, row, idx(width));
    x = arith::RemUIOp::create(b, loc, row, idx(width));
    memref::StoreOp::create(b, loc, value, output,
                            nchw ? ValueRange{zero, co, y, x}
                                 : ValueRange{zero, y, x, co});
    b.setInsertionPointAfter(copyRow);
    if (trace) {
      auto end = ::buddy::trace::EndOp::create(
          b, loc, output.getType(), output, cast<IntegerAttr>(trace.get("id")),
          cast<BoolAttr>(trace.get("cycle")),
          cast<BoolAttr>(trace.get("tensor")),
          cast<StringAttr>(trace.get("tag")));
      end->setAttr("id_path", trace.get("id_path"));
    }
    memref::DeallocOp::create(b, loc, patches);
    memref::DeallocOp::create(b, loc, weights);
    memref::DeallocOp::create(b, loc, result);
    op.erase();
  }
  return success();
}
