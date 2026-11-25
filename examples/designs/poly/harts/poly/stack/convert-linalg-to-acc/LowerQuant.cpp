//===- LowerQuant.cpp - Poly quantized values exposed to the hart --------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mxint8quantizeOps.h"
#include "transposeOps.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerPolyQuantizedOutputs(ModuleOp module) {
  const auto &target = target::getTarget();
  SmallVector<FP32ToMXINT8Op> ops;
  module.walk([&](FP32ToMXINT8Op op) {
    auto function = op->getParentOfType<func::FuncOp>();
    auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
    if (profile && profile.getValue() == target.hart &&
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
    if (axis == rank - 1 && k > 0 && k <= 65535 && k <= target.bankDepth * 4 &&
        k * 4 + 64 <= (target.dmaMaxRanges - 1) * 4096)
      chunk = k;
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
    Value storage, storageOffset;
    SmallVector<Value> storageStrides;
    if (direct) {
      storage = bufferization::ToBufferOp::create(
          b, loc,
          MemRefType::get(
              type.getShape(), b.getF32Type(),
              StridedLayoutAttr::get(
                  b.getContext(), ShapedType::kDynamic,
                  SmallVector<int64_t>(rank, ShapedType::kDynamic))),
          input, true);
      auto metadata = memref::ExtractStridedMetadataOp::create(b, loc, storage);
      storageOffset = metadata.getOffset();
      storageStrides = metadata.getStrides();
    } else
      storage = memref::AllocOp::create(
          b, loc, MemRefType::get({chunks, staticRows, chunk}, b.getF32Type()),
          dynamic);
    bool compact = axis == rank - 1 && (k < 0 || k + k / 32 <= 4095);
    bool packedDirect = axis == rank - 1 && chunks == 1 && packed == k + k / 32;
    Value quantized;
    if (!compact && !packedDirect)
      quantized = memref::AllocOp::create(
          b, loc, MemRefType::get({chunks, staticRows, packed}, b.getI8Type()),
          dynamic);
    auto destination = cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult().getUsers().begin());
    Value output = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(outType.getShape(), b.getI8Type()),
        destination.getDest());
    Value flatOutput;
    if (compact || packedDirect) {
      ReassociationIndices axes(rank);
      std::iota(axes.begin(), axes.end(), 0);
      flatOutput = memref::CollapseShapeOp::create(
          b, loc, output, ArrayRef<ReassociationIndices>{axes});
    }
    auto nest = [&](ValueRange sizes, bool stream,
                    function_ref<void(ValueRange)> body) {
      OpBuilder::InsertionGuard guard(b);
      SmallVector<Value> ivs;
      for (Value size : sizes) {
        auto loop = scf::ForOp::create(b, loc, c(0), size, c(1));
        if (stream)
          loop->setAttr("hart.stream", b.getUnitAttr());
        ivs.push_back(loop.getInductionVar());
        b.setInsertionPointToStart(loop.getBody());
      }
      body(ivs);
    };
    Value copyRows = rows, trailing, tailColumns, completeColumns;
    bool transposeInput = axis == rank - 2;
    if (transposeInput) {
      trailing = dims.back();
      tailColumns = b.createOrFold<arith::RemUIOp>(loc, trailing, c(4));
      completeColumns =
          b.createOrFold<arith::SubIOp>(loc, trailing, tailColumns);
      Value planes = c(1);
      for (int64_t dim = 0; dim < axis; ++dim)
        planes = b.createOrFold<arith::MulIOp>(loc, planes, dims[dim]);
      copyRows = b.createOrFold<arith::MulIOp>(loc, planes, tailColumns);
      Value memory = bufferization::ToBufferOp::create(
          b, loc, MemRefType::get(type.getShape(), b.getF32Type()), input,
          true);
      ReassociationIndices axes(rank);
      std::iota(axes.begin(), axes.end(), 0);
      Value flatInput = memref::CollapseShapeOp::create(
          b, loc, memory, ArrayRef<ReassociationIndices>{axes});
      Value flatStorage = memref::CollapseShapeOp::create(
          b, loc, storage, ArrayRef<ReassociationIndices>{{0, 1, 2}});
      Value in = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value groups = b.createOrFold<arith::DivUIOp>(loc, trailing, c(4));
      Value dmaRows = c(1);
      for (int64_t height = 2; height <= chunk; height *= 2) {
        int64_t maxStride = std::min<int64_t>(
            1023, ((target.dmaMaxRanges - 1) * 4096 - 80) / (4 * (height - 1)));
        Value fits = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule,
                                           trailing, c(maxStride));
        dmaRows = arith::SelectOp::create(b, loc, fits, c(height), dmaRows);
      }
      Value strided = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                            dmaRows, c(1));
      Value memoryStride = arith::SelectOp::create(
          b, loc, strided, arith::MulIOp::create(b, loc, trailing, c(4)),
          c(16));
      Value completeBlocks =
          b.createOrFold<arith::DivUIOp>(loc, dims[axis], c(chunk));
      Value tailK = b.createOrFold<arith::RemUIOp>(loc, dims[axis], c(chunk));
      for (bool partial : {false, true}) {
        if (partial && k > 0)
          continue;
        OpBuilder::InsertionGuard phase(b);
        if (partial) {
          Value nonempty = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ugt, tailK, c(0));
          auto active = scf::IfOp::create(b, loc, nonempty, false);
          b.setInsertionPointToStart(&active.getThenRegion().front());
          BankClearOp::create(b, loc, b.getI64Type(), in, 1, 1);
        }
        Value valid = partial ? tailK : c(chunk);
        Value loads = arith::DivUIOp::create(b, loc, valid, dmaRows);
        Value remainder = arith::RemUIOp::create(b, loc, valid, dmaRows);
        nest(
            {planes, partial ? c(1) : completeBlocks, groups}, true,
            [&](ValueRange at) {
              Value block = partial ? completeBlocks : at[1];
              Value column = arith::MulIOp::create(b, loc, at[2], c(4));
              Value firstK = arith::MulIOp::create(b, loc, block, c(chunk));
              Value base = arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, at[0], dims[axis]),
                  firstK);
              // Full row batches and the final incomplete batch use the same
              // bank. Clearing only the partial K block supplies its
              // quantization padding.
              for (bool tail : {false, true})
                nest({tail ? remainder : loads}, true, [&](ValueRange row) {
                  Value first =
                      tail ? arith::AddIOp::create(
                                 b, loc,
                                 arith::MulIOp::create(b, loc, loads, dmaRows),
                                 row[0])
                                 .getResult()
                           : arith::MulIOp::create(b, loc, row[0], dmaRows)
                                 .getResult();
                  Value offset = arith::AddIOp::create(
                      b, loc,
                      arith::MulIOp::create(
                          b, loc, arith::AddIOp::create(b, loc, base, first),
                          trailing),
                      column);
                  Value source = memref::SubViewOp::create(
                      b, loc, flatInput, ArrayRef<OpFoldResult>{offset},
                      ArrayRef<OpFoldResult>{b.getIndexAttr(4)},
                      ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                  BankMvinStridedOp::create(b, loc, b.getI64Type(), source, in,
                                            first, tail ? c(1) : dmaRows,
                                            tail ? c(16) : memoryStride, 0, 1,
                                            c(16), c(0));
                });
              Value result =
                  ::buddy::transpose_ball::BankTransposeF32Op::create(
                      b, loc, b.getI64Type(), in, out, chunk, 4);
              Value row = arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, at[0], trailing),
                  column);
              Value offset = arith::MulIOp::create(
                  b, loc, c(chunk),
                  arith::AddIOp::create(
                      b, loc, arith::MulIOp::create(b, loc, block, rows), row));
              Value resultMemory = memref::SubViewOp::create(
                  b, loc, flatStorage, ArrayRef<OpFoldResult>{offset},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(4 * chunk)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
              BankMvoutOp::create(b, loc, b.getI64Type(), resultMemory, result,
                                  chunk, 1);
            });
      }
      BankReleaseOp::create(b, loc, in);
      BankReleaseOp::create(b, loc, out);
    }
    if (axis == rank - 1 && k < 0) {
      Value memory = bufferization::ToBufferOp::create(
          b, loc, MemRefType::get(type.getShape(), b.getF32Type()), input,
          true);
      ReassociationIndices axes(rank);
      std::iota(axes.begin(), axes.end(), 0);
      Value flatInput = memref::CollapseShapeOp::create(
          b, loc, memory, ArrayRef<ReassociationIndices>{axes});
      Value flatStorage = memref::CollapseShapeOp::create(
          b, loc, storage, ArrayRef<ReassociationIndices>{{0, 1, 2}});
      Value complete = arith::DivUIOp::create(b, loc, dims[axis], c(chunk));
      Value tail = arith::RemUIOp::create(b, loc, dims[axis], c(chunk));
      Value words = arith::DivUIOp::create(b, loc, tail, c(4));
      Value remainder = arith::RemUIOp::create(b, loc, tail, c(4));
      Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      for (bool partial : {false, true}) {
        OpBuilder::InsertionGuard phase(b);
        if (partial) {
          Value nonempty = arith::CmpIOp::create(
              b, loc, arith::CmpIPredicate::ugt, tail, c(0));
          auto active = scf::IfOp::create(b, loc, nonempty, false);
          b.setInsertionPointToStart(&active.getThenRegion().front());
          BankClearOp::create(b, loc, b.getI64Type(), bank, 1, 1);
        }
        nest({partial ? c(1) : complete, rows}, true, [&](ValueRange at) {
          Value block = partial ? complete : at[0];
          Value offset = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, at[1], dims[axis]),
              arith::MulIOp::create(b, loc, block, c(chunk)));
          for (int piece = 0; piece < (partial ? 2 : 1); ++piece) {
            OpBuilder::InsertionGuard pieceInsertion(b);
            Value count = partial ? (piece ? remainder : words) : c(chunk / 4);
            if (partial) {
              Value nonempty = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::ugt, count, c(0));
              auto active = scf::IfOp::create(b, loc, nonempty, false);
              b.setInsertionPointToStart(&active.getThenRegion().front());
            }
            Value from = piece ? arith::AddIOp::create(
                                     b, loc, offset,
                                     arith::MulIOp::create(b, loc, words, c(4)))
                                     .getResult()
                               : offset;
            Value source = memref::SubViewOp::create(
                b, loc, flatInput, ArrayRef<OpFoldResult>{from},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            Value bytes =
                piece
                    ? arith::MulIOp::create(b, loc, remainder, c(4)).getResult()
                    : c(16);
            BankMvinStridedOp::create(
                b, loc, b.getI64Type(), source, bank, piece ? words : c(0),
                piece ? c(1) : count, c(16), 0, 1, bytes, c(0));
          }
          Value to = arith::MulIOp::create(
              b, loc, c(chunk),
              arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, block, rows), at[1]));
          Value result = memref::SubViewOp::create(
              b, loc, flatStorage, ArrayRef<OpFoldResult>{to},
              ArrayRef<OpFoldResult>{b.getIndexAttr(chunk)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          BankMvoutOp::create(b, loc, b.getI64Type(), result, bank, chunk / 4,
                              1);
        });
      }
      BankReleaseOp::create(b, loc, bank);
      copyRows = c(0);
    }
    // Contiguous rows already have the input layout the Ball consumes.
    if (!direct) {
      nest({chunkCount, copyRows, c(chunk)}, false, [&](ValueRange at) {
        SmallVector<Value> coords(rank);
        Value row = at[1];
        if (transposeInput)
          row = arith::AddIOp::create(
              b, loc,
              arith::MulIOp::create(
                  b, loc, arith::DivUIOp::create(b, loc, row, tailColumns),
                  trailing),
              arith::AddIOp::create(
                  b, loc, completeColumns,
                  arith::RemUIOp::create(b, loc, row, tailColumns)));
        Value flat = row;
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
        memref::StoreOp::create(b, loc, value, storage,
                                ValueRange{at[0], row, at[2]});
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
    if (compact && k < 0)
      limit = std::min<int64_t>(limit, 1 + (1023 - chunk / 16) / (packed / 16));
    if (compact && k > 0)
      limit = std::min<int64_t>(
          limit,
          1 + ((target.dmaMaxRanges - 1) * 4096 - chunk - 64) / (k + k / 32));
    if (packedDirect)
      limit = std::min<int64_t>(limit, ((target.dmaMaxRanges - 1) * 4096 - 64) /
                                           (chunk * 4));
    int64_t tile = 1;
    while (tile * 2 <= limit)
      tile *= 2;
    SmallVector<Value> tileAxes{chunkCount};
    if (direct)
      for (int64_t dim = 0; dim < rank - 2; ++dim)
        tileAxes.push_back(dims[dim]);
    Value tileRows = direct && rank > 1 ? dims[rank - 2] : rows;
    nest(tileAxes, true, [&](ValueRange block) {
      Value planeRow = c(0), inputPlaneOffset = storageOffset;
      if (direct)
        for (int64_t dim = 0; dim < rank - 2; ++dim) {
          planeRow = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, planeRow, dims[dim]),
              block[dim + 1]);
          inputPlaneOffset = arith::AddIOp::create(
              b, loc, inputPlaneOffset,
              arith::MulIOp::create(b, loc, block[dim + 1],
                                    storageStrides[dim]));
        }
      planeRow = arith::MulIOp::create(b, loc, planeRow, tileRows);
      Value remaining = tileRows, done = c(0);
      for (int64_t height = tile; height; height /= 2) {
        Value trips = b.createOrFold<arith::DivUIOp>(loc, remaining, c(height));
        nest({trips}, true, [&](ValueRange at) {
          Value firstInPlane = arith::AddIOp::create(
              b, loc, done, arith::MulIOp::create(b, loc, at[0], c(height)));
          Value first = arith::AddIOp::create(b, loc, planeRow, firstInPlane);
          Value line = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, block[0], rows), first);
          Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          Value q = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
          if (direct) {
            nest({c(height)}, true, [&](ValueRange row) {
              Value logical =
                  arith::AddIOp::create(b, loc, firstInPlane, row[0]);
              Value offset = arith::AddIOp::create(
                  b, loc, inputPlaneOffset,
                  arith::MulIOp::create(b, loc, logical,
                                        rank > 1 ? storageStrides[rank - 2]
                                                 : c(chunk)));
              Value source = matrix(storage, offset, 1, c(1), chunk, c(chunk));
              BankMvinStridedOp::create(
                  b, loc, b.getI64Type(), source, a,
                  arith::MulIOp::create(b, loc, row[0], c(chunk / 4)),
                  c(chunk / 4), c(16), 0, 1, c(16), c(0));
            });
          } else {
            Value source =
                matrix(storage, arith::MulIOp::create(b, loc, line, c(chunk)),
                       height, c(height), chunk, c(chunk));
            a = BankMvinOp::create(b, loc, b.getI64Type(), source, a,
                                   height * chunk / 4, 1);
          }
          q = ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
              b, loc, b.getI64Type(), a, q, height, chunk);
          if (packedDirect) {
            Value offset = arith::MulIOp::create(b, loc, first, c(packed));
            Value memory = memref::SubViewOp::create(
                b, loc, flatOutput, ArrayRef<OpFoldResult>{offset},
                ArrayRef<OpFoldResult>{b.getIndexAttr(height * packed)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            BankMvoutOp::create(b, loc, b.getI64Type(), memory, q,
                                height * packed / 16, 1);
          } else if (compact && (k < 0 || !llvm::isPowerOf2_64(chunk / 16))) {
            // Per-row transfers support dynamic memory strides and code spans
            // that cannot use the DMA power-of-two grouping field.
            nest({c(height)}, true, [&](ValueRange row) {
              Value outputRow = arith::AddIOp::create(b, loc, first, row[0]);
              for (int part = 0; part < 3; ++part) {
                int64_t count = part == 0   ? chunk / 16
                                : part == 1 ? chunk / 512
                                            : 1;
                int64_t bytes = part == 2 ? (chunk / 32) % 16 : 16;
                if (count == 0 || bytes == 0)
                  continue;
                int64_t scaleOffset = part == 2 ? chunk / 512 * 16 : 0;
                Value offset = arith::AddIOp::create(
                    b, loc, arith::MulIOp::create(b, loc, outputRow, packedK),
                    arith::AddIOp::create(
                        b, loc,
                        part ? arith::AddIOp::create(b, loc, paddedK,
                                                     c(scaleOffset))
                                   .getResult()
                             : c(0),
                        arith::MulIOp::create(b, loc, block[0],
                                              c(part ? chunk / 32 : chunk))));
                Value memory = memref::SubViewOp::create(
                    b, loc, flatOutput, ArrayRef<OpFoldResult>{offset},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(count * bytes)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
                Value bankRow = arith::AddIOp::create(
                    b, loc,
                    arith::MulIOp::create(b, loc, row[0], c(packed / 16)),
                    c(part ? chunk / 16 + scaleOffset / 16 : 0));
                BankMvoutStridedOp::create(b, loc, b.getI64Type(), memory, q,
                                           bankRow, c(count), c(16), 0, 1,
                                           c(bytes), c(0));
              }
            });
          } else if (compact) {
            // Codes and exponents land directly in their logical rows.
            // DMA skips bank padding, including partial exponent words.
            for (int part = 0; part < 2; ++part) {
              int64_t width = part ? chunk / 32 : chunk;
              Value offset = arith::AddIOp::create(
                  b, loc, arith::MulIOp::create(b, loc, first, packedK),
                  arith::AddIOp::create(
                      b, loc, part ? paddedK : c(0),
                      arith::MulIOp::create(b, loc, block[0], c(width))));
              Value memory = memref::SubViewOp::create(
                  b, loc, flatOutput, ArrayRef<OpFoldResult>{offset},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(width)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
              q = BankMvoutStridedOp::create(
                  b, loc, b.getI64Type(), memory, q, c(part ? chunk / 16 : 0),
                  c(height * (part ? 1 : chunk / 16)), packedK, 0, packed / 16,
                  c(part ? width : 16),
                  c(part ? 0 : llvm::Log2_64(chunk / 16)));
            }
          } else {
            Value result = matrix(
                quantized, arith::MulIOp::create(b, loc, line, c(packed)),
                height, c(height), packed, c(packed));
            BankMvoutOp::create(b, loc, b.getI64Type(), result, q,
                                height * packed / 16, 1);
          }
          BankReleaseOp::create(b, loc, a);
          BankReleaseOp::create(b, loc, q);
        });
        done = b.createOrFold<arith::AddIOp>(
            loc, done, b.createOrFold<arith::MulIOp>(loc, trips, c(height)));
        remaining = b.createOrFold<arith::RemUIOp>(loc, remaining, c(height));
      }
    });
    Value outputStride = c(1);
    for (int64_t dim = axis + 1; dim < rank; ++dim) {
      outputStride =
          b.createOrFold<arith::MulIOp>(loc, outputStride, dims[dim]);
    }
    if (!compact && !packedDirect && axis == rank - 1) {
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
          copy->setAttr("hart.after", b.getUnitAttr());
        }
      });
    } else if (!compact && !packedDirect) {
      Value planes = c(1);
      for (int64_t dim = 0; dim < axis; ++dim)
        planes = b.createOrFold<arith::MulIOp>(loc, planes, dims[dim]);
      int64_t scatterLimit = std::min<int64_t>(1023, target.bankDepth);
      scatterLimit = std::min<int64_t>(
          scatterLimit, 1 + ((target.dmaMaxRanges - 1) * 4096 - 65) / packed);
      int64_t scatterTile = 1;
      while (scatterTile * 2 <= scatterLimit)
        scatterTile *= 2;
      Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      nest({planes, chunkCount}, true, [&](ValueRange at) {
        for (int part = 0; part < 2; ++part) {
          int64_t count = part ? chunk / 32 : chunk;
          nest({c(count)}, true, [&](ValueRange byte) {
            Value axisIndex = arith::AddIOp::create(
                b, loc, part ? paddedK : c(0),
                arith::AddIOp::create(
                    b, loc, arith::MulIOp::create(b, loc, at[1], c(count)),
                    byte[0]));
            Value remaining = outputStride, done = c(0);
            for (int64_t height = scatterTile; height; height /= 2) {
              Value trips =
                  b.createOrFold<arith::DivUIOp>(loc, remaining, c(height));
              nest({trips}, true, [&](ValueRange tile) {
                Value first = arith::AddIOp::create(
                    b, loc, done,
                    arith::MulIOp::create(b, loc, tile[0], c(height)));
                Value row = arith::AddIOp::create(
                    b, loc, arith::MulIOp::create(b, loc, at[0], outputStride),
                    first);
                Value from = arith::AddIOp::create(
                    b, loc,
                    arith::MulIOp::create(
                        b, loc, c(packed),
                        arith::AddIOp::create(
                            b, loc, arith::MulIOp::create(b, loc, at[1], rows),
                            row)),
                    arith::AddIOp::create(b, loc, c(part ? chunk : 0),
                                          byte[0]));
                Value to = arith::AddIOp::create(
                    b, loc,
                    arith::MulIOp::create(
                        b, loc, outputStride,
                        arith::AddIOp::create(
                            b, loc,
                            arith::MulIOp::create(b, loc, at[0], packedK),
                            axisIndex)),
                    first);
                // Gather one byte from each packed row, then write adjacent
                // output columns. Both DMA strides are bounded constants.
                Value source =
                    matrix(quantized, from, height, c(height), 1, c(packed));
                Value result = matrix(output, to, height, c(height), 1, c(1));
                Value loaded = BankMvinStridedOp::create(
                    b, loc, b.getI64Type(), source, bank, c(0), c(height),
                    c(packed), 0, 1, c(1), c(0));
                BankMvoutStridedOp::create(b, loc, b.getI64Type(), result,
                                           loaded, c(0), c(height), c(1), 0, 1,
                                           c(1), c(0));
              });
              done = b.createOrFold<arith::AddIOp>(
                  loc, done,
                  b.createOrFold<arith::MulIOp>(loc, trips, c(height)));
              remaining =
                  b.createOrFold<arith::RemUIOp>(loc, remaining, c(height));
            }
          });
        }
      });
      BankReleaseOp::create(b, loc, bank);
    }
    Value written =
        bufferization::ToTensorOp::create(b, loc, outType, output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
