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
    if (k == 0 || (k > 0 && k % 32))
      return op.emitError("poly quantization needs positive K aligned to 32");
    int64_t chunk = k < 0 ? 32 : std::gcd(k, int64_t(256));
    int64_t chunks = k < 0 ? ShapedType::kDynamic : k / chunk;
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
    Value chunkCount = b.createOrFold<arith::DivUIOp>(
        loc, b.createOrFold<arith::AddIOp>(loc, dims[axis], c(chunk - 1)),
        c(chunk));
    Value paddedK = b.createOrFold<arith::MulIOp>(loc, chunkCount, c(chunk));
    Value packedK = b.createOrFold<arith::AddIOp>(
        loc, paddedK, b.createOrFold<arith::DivUIOp>(loc, paddedK, c(32)));
    SmallVector<Value> dynamic;
    if (chunks < 0)
      dynamic.push_back(chunkCount);
    if (staticRows < 0)
      dynamic.push_back(rows);
    bool direct = axis == rank - 1 && chunks == 1;
    Value storage;
    if (direct)
      storage = bufferization::ToBufferOp::create(
          b, loc, MemRefType::get(type.getShape(), b.getF32Type()), input,
          true);
    else
      storage = memref::AllocOp::create(
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
    // Contiguous rows already have the input layout the Ball consumes.
    if (!direct) {
      nest({chunkCount, rows, c(chunk)}, false, [&](ValueRange at) {
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
        Value value;
        if (k < 0) {
          Value valid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                              coords[axis], dims[axis]);
          auto choose =
              scf::IfOp::create(b, loc, TypeRange{b.getF32Type()}, valid, true);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&choose.getThenRegion().front());
          scf::YieldOp::create(
              b, loc,
              ValueRange{tensor::ExtractOp::create(b, loc, input, coords)});
          b.setInsertionPointToStart(&choose.getElseRegion().front());
          scf::YieldOp::create(b, loc,
                               ValueRange{arith::ConstantOp::create(
                                   b, loc, b.getF32FloatAttr(0))});
          value = choose.getResult(0);
        } else {
          value = tensor::ExtractOp::create(b, loc, input, coords);
        }
        memref::StoreOp::create(b, loc, value, storage, at);
      });
    }
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
    int64_t limit =
        std::min<int64_t>(1023, std::min(target.bankDepth / (chunk / 4),
                                         target.bankDepth / (packed / 16)));
    int64_t tile = 1;
    while (tile * 2 <= limit)
      tile *= 2;
    nest({chunkCount}, true, [&](ValueRange block) {
      Value remaining = rows, done = c(0);
      for (int64_t height = tile; height; height /= 2) {
        Value trips = b.createOrFold<arith::DivUIOp>(loc, remaining, c(height));
        nest({trips}, true, [&](ValueRange at) {
          Value first = arith::AddIOp::create(
              b, loc, done, arith::MulIOp::create(b, loc, at[0], c(height)));
          Value line = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, block[0], rows), first);
          Value source =
              matrix(storage, arith::MulIOp::create(b, loc, line, c(chunk)),
                     height, c(height), chunk, c(chunk));
          Value result =
              matrix(quantized, arith::MulIOp::create(b, loc, line, c(packed)),
                     height, c(height), packed, c(packed));
          Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          Value q = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          a = BankMvinOp::create(b, loc, b.getI64Type(), source, a,
                                 height * chunk / 4, 1);
          q = ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
              b, loc, b.getI64Type(), a, q, height, chunk);
          BankMvoutOp::create(b, loc, b.getI64Type(), result, q,
                              height * packed / 16, 1);
          BankReleaseOp::create(b, loc, a);
          BankReleaseOp::create(b, loc, q);
        });
        done = b.createOrFold<arith::AddIOp>(
            loc, done, b.createOrFold<arith::MulIOp>(loc, trips, c(height)));
        remaining = b.createOrFold<arith::RemUIOp>(loc, remaining, c(height));
      }
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
      nest({chunkCount}, false, [&](ValueRange at) {
        Value start = arith::MulIOp::create(
            b, loc, arith::MulIOp::create(b, loc, at[0], rows), c(packed));
        for (int part = 0; part < 2; ++part) {
          int64_t width = part ? chunk / 32 : chunk;
          Value from =
              arith::AddIOp::create(b, loc, start, c(part ? chunk : 0));
          Value to = arith::AddIOp::create(
              b, loc, (part ? paddedK : c(0)),
              arith::MulIOp::create(b, loc, at[0], c(width)));
          auto copy = memref::CopyOp::create(
              b, loc,
              matrix(quantized, from, staticRows, rows, width, c(packed)),
              matrix(output, to, staticRows, rows, width, packedK));
          copy->setAttr("core.after", b.getUnitAttr());
        }
      });
    } else {
      nest({rows, chunkCount}, false, [&](ValueRange at) {
        Value flat = at[0], base = c(0), stride = c(1);
        for (int64_t dim = rank - 1; dim >= 0; --dim) {
          if (dim != axis) {
            Value coord = b.createOrFold<arith::RemUIOp>(loc, flat, dims[dim]);
            flat = b.createOrFold<arith::DivUIOp>(loc, flat, dims[dim]);
            base = arith::AddIOp::create(
                b, loc, base, arith::MulIOp::create(b, loc, coord, stride));
          }
          stride = arith::MulIOp::create(b, loc, stride,
                                         dim == axis ? packedK : dims[dim]);
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
                      b, loc, (part ? paddedK : c(0)),
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
