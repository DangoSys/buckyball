//===- LowerRowOperators.cpp - row_operator on the GP domain
//------------------===//
//
// A row_operator becomes one launch of its operator per chunk of a row (its
// `chunk` columns of the result, the row's inputs in proportion), or per
// `rows` whole rows along the axis before the row (each input those rows,
// contiguous or apart along its own axis, or one row broadcast along them,
// by one DMA): the CPU
// prepares the launch descriptors of the rows (entry, code bytes, stack, the
// arguments, a7 the operator number), then one affine row loop moves, for each
// chunk, the input chunks into the groups of the read bank and the
// descriptor into the first group of the write bank, runs the operator and
// moves the chunk out. The operators are one image,
// `operators.image` linked in as _binary_operators_image_start, loaded into the
// program bank of the subgraph.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::acc;

MemRefType bufferType(Value tensor);

// Loads the operator image into bank `program`: its code bytes and entry.
std::pair<Value, Value> loadOperators(OpBuilder &b, Location loc,
                                      int64_t program) {
  auto module = b.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  auto imageType = MemRefType::get({1}, b.getI8Type());
  StringRef imageName = "_binary_operators_image_start";
  if (!module.lookupSymbol<memref::GlobalOp>(imageName)) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    memref::GlobalOp::create(b, loc, imageName, StringAttr(), imageType,
                             Attribute(), true, IntegerAttr());
  }
  Value image = memref::GetGlobalOp::create(b, loc, imageType, imageName);
  // The header: magic, code bytes, entry, constant base, constant bytes, 0.
  Value header =
      memref::ViewOp::create(b, loc, MemRefType::get({6}, b.getI32Type()),
                             image, index(0), ValueRange{});
  auto word = [&](int64_t at) -> Value {
    return arith::ExtUIOp::create(
        b, loc, b.getI64Type(),
        memref::LoadOp::create(b, loc, header, ValueRange{index(at)}));
  };
  Value codeSize = word(1), entry = word(2);
  Value totalSize = arith::AddIOp::create(
      b, loc, arith::ConstantIntOp::create(b, loc, 24, 64),
      arith::AddIOp::create(b, loc, codeSize, word(4)));
  MvinOperatorOp::create(b, loc, image, totalSize, program);
  return {codeSize, entry};
}

LogicalResult lowerRowOperators(ModuleOp module,
                                function_ref<int64_t(StringAttr)> number) {
  SmallVector<CustomOp> calls;
  module.walk([&](CustomOp op) {
    if (op.getKind() == "row_operator")
      calls.push_back(op);
  });
  for (CustomOp op : calls) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t width = type.getShape().back();
    int64_t chunk = op->getAttrOfType<IntegerAttr>("chunk").getInt();
    int64_t batch = op->getAttrOfType<IntegerAttr>("rows").getInt();
    // The axes of the launch loops: all but the row, and the batched one.
    int64_t loopAxes = type.getRank() - (batch > 1 ? 2 : 1);
    int64_t chunks = width / chunk;
    if (!target::getTarget().rvvEnabled ||
        (chunk + 3) / 4 > target::getTarget().bankDepth)
      return op.emitError("row exceeds the GP domain or the banks");
    auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult(0).getUsers().begin());
    if (!destination || !op.getResult(0).hasOneUse())
      return op.emitError("row_operator must write its hart destination");
    int64_t program = op->getParentOfType<func::FuncOp>()
                          ->getAttrOfType<IntegerAttr>("hart.program_bank")
                          .getInt();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    auto integer = [&](uint64_t value) -> Value {
      return arith::ConstantIntOp::create(b, loc, value, 64);
    };
    auto [codeSize, entry] = loadOperators(b, loc, program);
    // A row of `width` elements (FP32, or the bytes of a mask) between
    // memory and a bank: whole 16-byte rows, and a row of the remaining
    // bytes by strided DMA (a row need not start on 16 bytes then).
    auto move = [&](Value row, Value bank, int64_t width, bool out) -> Value {
      auto rowType = cast<MemRefType>(row.getType());
      int64_t lanes = 16 / ((rowType.getElementTypeBitWidth() + 7) / 8);
      if (width % lanes == 0)
        return out ? Value(BankMvoutOp::create(b, loc, b.getI64Type(), row,
                                               bank, width / lanes, 1))
                   : Value(BankMvinOp::create(b, loc, b.getI64Type(), row, bank,
                                              width / lanes, 1));
      auto part = [&](int64_t first, int64_t count) {
        SmallVector<OpFoldResult> at(rowType.getRank(), b.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes(rowType.getRank(), b.getIndexAttr(1));
        SmallVector<OpFoldResult> strides(rowType.getRank(), b.getIndexAttr(1));
        at.back() = b.getIndexAttr(first);
        sizes.back() = b.getIndexAttr(count);
        return memref::SubViewOp::create(b, loc, row, at, sizes, strides)
            .getResult();
      };
      auto strided = [&](Value memory, int64_t first, int64_t rows,
                         int64_t bytes) {
        bank = out ? Value(BankMvoutStridedOp::create(
                         b, loc, b.getI64Type(), memory, bank, index(first),
                         index(rows), index(16), 0, 1, index(bytes), index(0)))
                   : Value(BankMvinStridedOp::create(
                         b, loc, b.getI64Type(), memory, bank, index(first),
                         index(rows), index(16), 0, 1, index(bytes), index(0)));
      };
      if (width / lanes)
        strided(part(0, width / lanes * lanes), 0, width / lanes, 16);
      strided(part(width / lanes * lanes, width % lanes), width / lanes, 1,
              width % lanes * (16 / lanes));
      return bank;
    };
    int64_t operatorIndex =
        number(op->getAttrOfType<FlatSymbolRefAttr>("operator").getAttr());
    auto maps = op->getAttrOfType<ArrayAttr>("maps");
    auto inputOffsets = op->getAttrOfType<DenseI64ArrayAttr>("input_offsets");
    auto inputWidths = op->getAttrOfType<DenseI64ArrayAttr>("input_widths");
    SmallVector<bool> scalar;
    int64_t vectors = 0, rows = type.getNumElements() / width;
    SmallVector<Value> inputs;
    DenseMap<Value, Value> buffers;
    for (auto [i, input] : llvm::enumerate(op.getInputs())) {
      auto inputType = cast<RankedTensorType>(input.getType());
      AffineMap map = cast<AffineMapAttr>(maps[i]).getValue();
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      bool vector =
          isa<AffineDimExpr>(last) &&
          cast<AffineDimExpr>(last).getPosition() == type.getRank() - 1;
      scalar.push_back(!vector);
      vectors += vector;
      // One read-only buffer of each value: a value read as several inputs
      // (the rows of a grouped reduction) is not copied.
      Value &read = buffers[input];
      if (!read) {
        read =
            bufferization::ToBufferOp::create(b, loc, bufferType(input), input);
        read.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      }
      inputs.push_back(read);
    }
    int64_t readGroups = std::max<int64_t>(1, vectors);
    // The descriptors differ only in the scalar inputs: without them one
    // descriptor serves every row.
    bool shared = llvm::none_of(scalar, [](bool s) { return s; });
    Value descriptors = memref::AllocOp::create(
        b, loc, MemRefType::get({shared ? 1 : rows, 12}, b.getI64Type()));
    descriptors.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
    Value output = bufferization::ToBufferOp::create(
        b, loc, bufferType(destination.getDest()), destination.getDest());
    SmallVector<Value> coordinates;
    Operation *outer = nullptr;
    for (int64_t dimension : type.getShape().drop_back()) {
      if (shared) {
        coordinates.push_back(index(0));
        continue;
      }
      auto loop =
          scf::ForOp::create(b, loc, index(0), index(dimension), index(1));
      if (!outer)
        outer = loop;
      coordinates.push_back(loop.getInductionVar());
      b.setInsertionPointToStart(loop.getBody());
    }
    coordinates.push_back(index(0));
    Value row = index(0);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      row = arith::AddIOp::create(
          b, loc,
          arith::MulIOp::create(b, loc, row, index(type.getDimSize(axis))),
          coordinates[axis]);
    auto store = [&](int64_t field, Value value) {
      memref::StoreOp::create(b, loc, value, descriptors,
                              ValueRange{row, index(field)});
    };
    store(0, entry);
    store(1, codeSize);
    store(2, integer(0x40002000));
    for (int64_t field = 3; field < 12; ++field)
      store(field, integer(0));
    store(10, integer(operatorIndex));
    int64_t group = 0;
    for (auto [i, input] : llvm::enumerate(inputs)) {
      AffineMap map = cast<AffineMapAttr>(maps[i]).getValue();
      SmallVector<Value> offsets;
      for (AffineExpr expression : map.getResults())
        offsets.push_back(
            isa<AffineDimExpr>(expression)
                ? coordinates[cast<AffineDimExpr>(expression).getPosition()]
                : index(0));
      offsets.back() =
          arith::AddIOp::create(b, loc, offsets.back(), index(inputOffsets[i]));
      if (scalar[i]) {
        Value value = memref::LoadOp::create(b, loc, input, offsets);
        store(3 + i,
              arith::ExtUIOp::create(
                  b, loc, b.getI64Type(),
                  arith::BitcastOp::create(b, loc, b.getI32Type(), value)));
        continue;
      }
      store(3 + i, integer(uint64_t(group++) << 32));
    }
    store(3 + inputs.size(), integer(uint64_t(readGroups) << 32));
    if (outer)
      b.setInsertionPointAfter(outer);

    // Prepare descriptors on the CPU, then record one affine row loop.
    // Its command count is independent of the number of tensor rows.
    coordinates.clear();
    outer = nullptr;
    for (int64_t dimension : type.getShape().take_front(loopAxes)) {
      auto loop =
          scf::ForOp::create(b, loc, index(0), index(dimension), index(1));
      loop->setAttr("hart.stream", b.getUnitAttr());
      if (!outer)
        outer = loop;
      coordinates.push_back(loop.getInductionVar());
      b.setInsertionPointToStart(loop.getBody());
    }
    if (batch > 1) {
      // The launches along the batched axis, `batch` rows each.
      int64_t launches = type.getDimSize(type.getRank() - 2) / batch;
      if (launches * batch != type.getDimSize(type.getRank() - 2))
        return op.emitError("rows of a launch do not divide their axis");
      auto loop =
          scf::ForOp::create(b, loc, index(0), index(launches), index(1));
      loop->setAttr("hart.stream", b.getUnitAttr());
      if (!outer)
        outer = loop;
      b.setInsertionPointToStart(loop.getBody());
      coordinates.push_back(
          arith::MulIOp::create(b, loc, loop.getInductionVar(), index(batch)));
    }
    coordinates.push_back(index(0));
    row = index(0);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      row = arith::AddIOp::create(
          b, loc,
          arith::MulIOp::create(b, loc, row, index(type.getDimSize(axis))),
          coordinates[axis]);
    Value descriptor = memref::SubViewOp::create(
        b, loc, descriptors,
        ArrayRef<OpFoldResult>{shared ? index(0) : row, b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(12)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    for (int64_t part = 0; part < chunks; ++part) {
      Value readBank;
      for (auto [i, input] : llvm::enumerate(inputs)) {
        if (scalar[i])
          continue;
        AffineMap map = cast<AffineMapAttr>(maps[i]).getValue();
        SmallVector<Value> offsets;
        for (AffineExpr expression : map.getResults())
          offsets.push_back(
              isa<AffineDimExpr>(expression)
                  ? coordinates[cast<AffineDimExpr>(expression).getPosition()]
                  : index(0));
        int64_t inputWidth = inputWidths[i] / chunks;
        SmallVector<OpFoldResult> at(offsets.begin(), offsets.end());
        at.back() =
            arith::AddIOp::create(b, loc, offsets.back(),
                                  index(inputOffsets[i] + part * inputWidth))
                .getResult();
        SmallVector<OpFoldResult> sizes(offsets.size(), b.getIndexAttr(1));
        SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
        // The input follows the batched rows, along its axis `axis`, or is
        // one row broadcast along them.
        int64_t axis = -1;
        if (batch > 1)
          for (int64_t result = 0; result + 1 < map.getNumResults(); ++result)
            if (map.getResult(result) ==
                getAffineDimExpr(type.getRank() - 2, op.getContext()))
              axis = result;
        bool along = axis >= 0;
        int64_t count = along ? batch : 1;
        if ((count * inputWidth + 3) / 4 > target::getTarget().bankDepth)
          return op.emitError("row_operator input row exceeds its bank");
        sizes.back() = b.getIndexAttr(inputWidth);
        if (along)
          sizes[axis] = b.getIndexAttr(batch);
        Value tile =
            memref::SubViewOp::create(b, loc, input, at, sizes, strides);
        Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
        // The elements from a row to the next along the axis, in the layout
        // of the input (a slice keeps the strides of its source).
        auto inputMemory = cast<MemRefType>(input.getType());
        int64_t stride = along ? inputMemory.getStridesAndOffset().first[axis]
                               : inputMemory.getShape().back();
        if (count == 1 || stride == inputWidth) {
          bank = move(tile, bank, count * inputWidth, false);
        } else {
          // Rows of `blocks` 16-byte rows `stride` elements apart.
          int64_t blocks = inputWidth / 4;
          bank = BankMvinStridedOp::create(
              b, loc, b.getI64Type(), tile, bank, index(0),
              index(batch * blocks), index(stride * 4), 0, blocks, index(16),
              index(llvm::Log2_64(blocks)));
        }
        readBank = readBank ? Value(BankTransferOp::create(
                                  b, loc, b.getI64Type(), bank, readBank))
                            : bank;
      }
      if (!readBank)
        readBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value writeBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      writeBank = BankMvinOp::create(b, loc, b.getI64Type(), descriptor,
                                     writeBank, 6, 1);
      writeBank = BankRunOperatorOp::create(b, loc, b.getI64Type(), readBank,
                                            writeBank, program);
      SmallVector<OpFoldResult> at(coordinates.begin(), coordinates.end());
      at.back() = b.getIndexAttr(part * chunk);
      SmallVector<OpFoldResult> sizes(type.getRank(), b.getIndexAttr(1));
      SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
      sizes.back() = b.getIndexAttr(chunk);
      if (batch > 1)
        sizes[sizes.size() - 2] = b.getIndexAttr(batch);
      Value result =
          memref::SubViewOp::create(b, loc, output, at, sizes, strides);
      writeBank = move(result, writeBank, batch * chunk, true);
      BankReleaseOp::create(b, loc, readBank);
      BankReleaseOp::create(b, loc, writeBank);
    }
    if (outer)
      b.setInsertionPointAfter(outer);
    MsetOp::create(b, loc, program, false, 0, 0);
    Value written =
        bufferization::ToTensorOp::create(b, loc, type, output, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
