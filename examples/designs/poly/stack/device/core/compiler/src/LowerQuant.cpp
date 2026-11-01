//===- LowerQuant.cpp - Poly quantized values exposed to the core --------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mxint8quantizeOps.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerPolyQuantizedOutputs(ModuleOp module) {
  const auto &target = buckyball_target::getBuckyballTarget();
  SmallVector<FP32ToMXINT8Op> ops;
  module.walk([&](FP32ToMXINT8Op op) {
    auto function = op->getParentOfType<func::FuncOp>();
    auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
    if (profile && profile.getValue() == target.core &&
        op.getResult().hasOneUse() &&
        isa<bufferization::MaterializeInDestinationOp>(
            *op.getResult().getUsers().begin()))
      ops.push_back(op);
  });
  for (auto op : ops) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto input = op.getInput();
    auto type = input.getType();
    auto outType = op.getResult().getType();
    int64_t axis = op.getAxis(), rank = type.getRank(),
            k = type.getDimSize(axis);
    if (k < 0 || k % 32)
      return op.emitError("poly quantization needs a static K multiple of 32");
    int64_t chunk = std::gcd(k, int64_t(256)), chunks = k / chunk;
    int64_t packed = ((chunk + chunk / 32 + 15) / 16) * 16;
    if (chunk / 4 > target.bankDepth)
      return op.emitError("poly quantization exceeds the input bank");
    auto c = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    Value rows = c(1);
    int64_t staticRows = 1;
    SmallVector<Value> dims;
    for (int64_t dim = 0; dim < rank; ++dim) {
      Value size = tensor::DimOp::create(b, loc, input, dim);
      dims.push_back(size);
      if (dim == axis)
        continue;
      rows = b.createOrFold<arith::MulIOp>(loc, rows, size);
      staticRows = staticRows < 0 || type.isDynamicDim(dim)
                       ? ShapedType::kDynamic
                       : staticRows * type.getDimSize(dim);
    }
    SmallVector<Value> dynamic;
    if (staticRows < 0)
      dynamic.push_back(rows);
    Value storage = memref::AllocOp::create(
        b, loc, MemRefType::get({chunks, staticRows, chunk}, b.getF32Type()),
        dynamic);
    Value quantized = memref::AllocOp::create(
        b, loc, MemRefType::get({chunks, staticRows, packed}, b.getI8Type()),
        dynamic);
    auto nest = [&](ValueRange sizes, bool stream,
                    function_ref<void(ValueRange)> body) {
      OpBuilder::InsertionGuard guard(b);
      SmallVector<Value> ivs;
      for (Value size : sizes) {
        auto loop = scf::ForOp::create(b, loc, c(0), size, c(1));
        if (stream)
          loop->setAttr("core.stream", b.getUnitAttr());
        ivs.push_back(loop.getInductionVar());
        b.setInsertionPointToStart(loop.getBody());
      }
      body(ivs);
    };
    // The host transposes only when the quantization axis is not contiguous.
    nest({c(chunks), rows, c(chunk)}, false, [&](ValueRange at) {
      SmallVector<Value> coords(rank);
      Value flat = at[1];
      for (int64_t dim = rank - 1; dim >= 0; --dim) {
        if (dim == axis)
          continue;
        coords[dim] = b.createOrFold<arith::RemUIOp>(loc, flat, dims[dim]);
        flat = b.createOrFold<arith::DivUIOp>(loc, flat, dims[dim]);
      }
      coords[axis] = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, at[0], c(chunk)), at[2]);
      memref::StoreOp::create(b, loc,
                              tensor::ExtractOp::create(b, loc, input, coords),
                              storage, at);
    });
    auto matrix = [&](Value memory, Value offset, int64_t height, Value extent,
                      int64_t width, Value stride) -> Value {
      Type element = cast<MemRefType>(memory.getType()).getElementType();
      APInt fixed;
      bool known = matchPattern(stride, m_ConstantInt(&fixed));
      int64_t rowStride = known ? fixed.getSExtValue() : ShapedType::kDynamic;
      OpFoldResult step = known ? OpFoldResult(b.getIndexAttr(rowStride))
                                : OpFoldResult(stride);
      return memref::ReinterpretCastOp::create(
          b, loc,
          MemRefType::get({height, width}, element,
                          StridedLayoutAttr::get(b.getContext(),
                                                 ShapedType::kDynamic,
                                                 {rowStride, 1})),
          memory, offset,
          ArrayRef<OpFoldResult>{height < 0
                                     ? OpFoldResult(extent)
                                     : OpFoldResult(b.getIndexAttr(height)),
                                 b.getIndexAttr(width)},
          ArrayRef<OpFoldResult>{step, b.getIndexAttr(1)});
    };
    nest({c(chunks), rows}, true, [&](ValueRange at) {
      Value line = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, at[0], rows), at[1]);
      Value source =
          matrix(storage, arith::MulIOp::create(b, loc, line, c(chunk)), 1,
                 c(1), chunk, c(chunk));
      Value result =
          matrix(quantized, arith::MulIOp::create(b, loc, line, c(packed)), 1,
                 c(1), packed, c(packed));
      Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value q = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      a = BankMvinOp::create(b, loc, b.getI64Type(), source, a, chunk / 4, 1);
      q = ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
          b, loc, b.getI64Type(), a, q, 1, chunk);
      BankMvoutOp::create(b, loc, b.getI64Type(), result, q, packed / 16, 1);
      BankReleaseOp::create(b, loc, a);
      BankReleaseOp::create(b, loc, q);
    });
    auto destination = cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult().getUsers().begin());
    Value output = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(outType.getShape(), b.getI8Type()),
        destination.getDest());
    Value outputStride = c(1);
    for (int64_t dim = axis + 1; dim < rank; ++dim) {
      outputStride =
          b.createOrFold<arith::MulIOp>(loc, outputStride, dims[dim]);
    }
    if (axis == rank - 1) {
      nest({c(chunks)}, false, [&](ValueRange at) {
        Value start = arith::MulIOp::create(
            b, loc, arith::MulIOp::create(b, loc, at[0], rows), c(packed));
        for (int part = 0; part < 2; ++part) {
          int64_t width = part ? chunk / 32 : chunk;
          Value from =
              arith::AddIOp::create(b, loc, start, c(part ? chunk : 0));
          Value to = arith::AddIOp::create(
              b, loc, c(part ? k : 0),
              arith::MulIOp::create(b, loc, at[0], c(width)));
          auto copy = memref::CopyOp::create(
              b, loc,
              matrix(quantized, from, staticRows, rows, width, c(packed)),
              matrix(output, to, staticRows, rows, width, c(k + k / 32)));
          copy->setAttr("core.after", b.getUnitAttr());
        }
      });
    } else {
      nest({rows, c(chunks)}, false, [&](ValueRange at) {
        Value flat = at[0], base = c(0), stride = c(1);
        for (int64_t dim = rank - 1; dim >= 0; --dim) {
          if (dim != axis) {
            Value coord = b.createOrFold<arith::RemUIOp>(loc, flat, dims[dim]);
            flat = b.createOrFold<arith::DivUIOp>(loc, flat, dims[dim]);
            base = arith::AddIOp::create(
                b, loc, base, arith::MulIOp::create(b, loc, coord, stride));
          }
          stride = arith::MulIOp::create(
              b, loc, stride, dim == axis ? c(k + k / 32) : dims[dim]);
        }
        Value start = arith::MulIOp::create(
            b, loc, c(packed),
            arith::AddIOp::create(
                b, loc, arith::MulIOp::create(b, loc, at[1], rows), at[0]));
        for (int part = 0; part < 2; ++part) {
          int64_t count = part ? chunk / 32 : chunk;
          Value from =
              arith::AddIOp::create(b, loc, start, c(part ? chunk : 0));
          Value to = arith::AddIOp::create(
              b, loc, base,
              arith::MulIOp::create(
                  b, loc, outputStride,
                  arith::AddIOp::create(
                      b, loc, c(part ? k : 0),
                      arith::MulIOp::create(b, loc, at[1], c(count)))));
          auto copy = memref::CopyOp::create(
              b, loc, matrix(quantized, from, count, c(count), 1, c(1)),
              matrix(output, to, count, c(count), 1, outputStride));
          copy->setAttr("core.after", b.getUnitAttr());
        }
      });
    }
    Value written =
        bufferization::ToTensorOp::create(b, loc, outType, output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
