//===- LowerRowOperators.cpp - row_operator on the GP domain
//------------------===//
//
// A row_operator becomes one launch of its operator per row: the CPU prepares
// the launch descriptors (entry, code bytes, stack, the arguments, a7 the
// operator number), then one affine row loop moves the input rows into the
// groups of the read bank and the descriptor into the first group of the write
// bank, runs the operator and moves the row out. The operators are one image,
// `operators.image` linked in as _binary_operators_image_start, loaded into the
// program bank of the subgraph.
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace ::buddy::acc;

// Loads the operator image into bank `program`: its code bytes and entry.
std::pair<Value, Value> loadOperators(OpBuilder &b, Location loc,
                                      int64_t program) {
  auto module = b.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  auto imageType = MemRefType::get({1}, b.getI8Type());
  StringRef imageName = target::getTarget().hart == "reduce"
                            ? "_binary_reduce_operators_image_start"
                            : "_binary_operators_image_start";
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
    auto function = op->getParentOfType<func::FuncOp>();
    auto profile = function->getAttrOfType<StringAttr>("bbacc.target");
    if (!profile || profile.getValue() != target::getTarget().hart)
      continue;
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t width = type.getShape().back();
    if (!target::getTarget().rvvEnabled ||
        (width + 3) / 4 > target::getTarget().bankDepth)
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
    int64_t operatorIndex =
        number(op->getAttrOfType<FlatSymbolRefAttr>("operator").getAttr());
    auto maps = op->getAttrOfType<ArrayAttr>("maps");
    auto inputOffsets = op->getAttrOfType<DenseI64ArrayAttr>("input_offsets");
    auto inputWidths = op->getAttrOfType<DenseI64ArrayAttr>("input_widths");
    SmallVector<bool> scalar;
    Value rows = index(1);
    SmallVector<Value> dimensions;
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis) {
      Value size = tensor::DimOp::create(b, loc, destination.getDest(), axis);
      dimensions.push_back(size);
      rows = b.createOrFold<arith::MulIOp>(loc, rows, size);
    }
    SmallVector<Value> inputs;
    for (auto [i, input] : llvm::enumerate(op.getInputs())) {
      auto inputType = cast<RankedTensorType>(input.getType());
      AffineMap map = cast<AffineMapAttr>(maps[i]).getValue();
      AffineExpr last = map.getResult(map.getNumResults() - 1);
      bool vector =
          isa<AffineDimExpr>(last) &&
          cast<AffineDimExpr>(last).getPosition() == type.getRank() - 1;
      scalar.push_back(!vector);
      SmallVector<int64_t> strides(inputType.getRank(), ShapedType::kDynamic);
      strides.back() = 1;
      auto layout =
          StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, strides);
      inputs.push_back(bufferization::ToBufferOp::create(
          b, loc,
          MemRefType::get(inputType.getShape(), inputType.getElementType(),
                          layout),
          input, true));
    }
    SmallVector<int64_t> groups(inputs.size(), -1);
    int64_t vectors = 0;
    for (int64_t i = 0; i < int64_t(inputs.size()); ++i) {
      if (scalar[i])
        continue;
      int64_t previous = 0;
      while (previous < i && (op.getInputs()[previous] != op.getInputs()[i] ||
                              maps[previous] != maps[i] ||
                              inputOffsets[previous] != inputOffsets[i] ||
                              inputWidths[previous] != inputWidths[i]))
        ++previous;
      groups[i] = previous < i ? groups[previous] : vectors++;
    }
    int64_t readGroups = std::max<int64_t>(1, vectors);
    bool varyingDescriptor = false;
    for (auto [i, attr] : llvm::enumerate(maps.getValue().drop_back()))
      if (scalar[i])
        varyingDescriptor |=
            llvm::any_of(cast<AffineMapAttr>(attr).getValue().getResults(),
                         [](AffineExpr expression) {
                           return isa<AffineDimExpr>(expression);
                         });
    Value descriptors = memref::AllocOp::create(
        b, loc, MemRefType::get({ShapedType::kDynamic, 12}, b.getI64Type()),
        ValueRange{varyingDescriptor ? rows : index(1)});
    descriptors.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
    SmallVector<int64_t> outputStrides(type.getRank(), ShapedType::kDynamic);
    outputStrides.back() = 1;
    bool borrowed = isa<BlockArgument>(destination.getDest());
    auto outputType = MemRefType::get(type.getShape(), b.getF32Type());
    if (borrowed)
      outputType = MemRefType::get(type.getShape(), b.getF32Type(),
                                   StridedLayoutAttr::get(b.getContext(),
                                                          ShapedType::kDynamic,
                                                          outputStrides));
    Value output = bufferization::ToBufferOp::create(b, loc, outputType,
                                                     destination.getDest());
    Value outputWidth =
        memref::DimOp::create(b, loc, output, type.getRank() - 1);
    SmallVector<Value> widths;
    for (auto [i, input] : llvm::enumerate(inputs)) {
      auto inputType = cast<MemRefType>(input.getType());
      Value extent = inputWidths[i] < 0
                         ? Value(memref::DimOp::create(b, loc, input,
                                                       inputType.getRank() - 1))
                         : index(inputWidths[i]);
      widths.push_back(extent);
      if (inputWidths[i] < 0)
        cf::AssertOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ule, extent,
                                  index(target::getTarget().bankDepth * 4)),
            "GP row exceeds bank capacity");
    }
    // Partial rows transfer only their valid bytes, including packed scales.
    auto transfer = [&](Value memory, Value bank, Value extent, bool load) {
      auto memoryType = cast<MemRefType>(memory.getType());
      Value flat = memory;
      if (memoryType.getRank() > 1) {
        ReassociationIndices axes;
        for (int64_t axis = 0; axis < memoryType.getRank(); ++axis)
          axes.push_back(axis);
        flat = memref::CollapseShapeOp::create(
            b, loc, memory, ArrayRef<ReassociationIndices>{axes});
      }
      int64_t elementBytes =
          memoryType.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t elementsPerRow = 16 / elementBytes;
      Value full =
          arith::DivUIOp::create(b, loc, extent, index(elementsPerRow));
      Value tail =
          arith::RemUIOp::create(b, loc, extent, index(elementsPerRow));
      for (bool partial : {false, true}) {
        Value count = partial ? tail : full;
        auto condition = scf::IfOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, count,
                                  index(0)),
            false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&condition.getThenRegion().front());
        Value offset = partial ? Value(arith::MulIOp::create(
                                     b, loc, full, index(elementsPerRow)))
                               : index(0);
        Value elements = partial ? tail
                                 : Value(arith::MulIOp::create(
                                       b, loc, full, index(elementsPerRow)));
        Value view = memref::SubViewOp::create(
            b, loc, flat, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{elements},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
        Value first = partial ? full : index(0);
        Value depth = partial ? index(1) : full;
        Value bytes = partial ? Value(arith::MulIOp::create(
                                    b, loc, tail, index(elementBytes)))
                              : index(16);
        if (load)
          BankMvinStridedOp::create(b, loc, b.getI64Type(), view, bank, first,
                                    depth, index(16), 0, 1, bytes, index(0));
        else
          BankMvoutStridedOp::create(b, loc, b.getI64Type(), view, bank, first,
                                     depth, index(16), 0, 1, bytes, index(0));
      }
    };
    SmallVector<Value> coordinates;
    Operation *outer = nullptr;
    for (Value dimension : dimensions) {
      if (!varyingDescriptor) {
        coordinates.push_back(index(0));
        continue;
      }
      auto loop = scf::ForOp::create(b, loc, index(0), dimension, index(1));
      if (!outer)
        outer = loop;
      coordinates.push_back(loop.getInductionVar());
      b.setInsertionPointToStart(loop.getBody());
    }
    coordinates.push_back(index(0));
    Value row = index(0);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      row = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, row, dimensions[axis]),
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
    if (auto widthInput = op->getAttrOfType<IntegerAttr>("width_input"))
      store(9, arith::IndexCastOp::create(b, loc, b.getI64Type(),
                                          widths[widthInput.getInt()]));
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
        if (value.getType().isF32())
          value = arith::BitcastOp::create(b, loc, b.getI32Type(), value);
        store(3 + i, arith::ExtUIOp::create(b, loc, b.getI64Type(), value));
        continue;
      }
      store(3 + i, integer(uint64_t(groups[i]) << 32));
    }
    store(3 + inputs.size(), integer(uint64_t(readGroups) << 32));
    if (outer)
      b.setInsertionPointAfter(outer);

    // Prepare row-dependent descriptors or one shared descriptor, then record
    // the row loop. Its command count is independent of the number of tensor
    // rows.
    coordinates.clear();
    outer = nullptr;
    for (Value dimension : dimensions) {
      auto loop = scf::ForOp::create(b, loc, index(0), dimension, index(1));
      loop->setAttr("hart.stream", b.getUnitAttr());
      if (!outer)
        outer = loop;
      coordinates.push_back(loop.getInductionVar());
      b.setInsertionPointToStart(loop.getBody());
    }
    coordinates.push_back(index(0));
    row = index(0);
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis)
      row = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, row, dimensions[axis]),
          coordinates[axis]);
    Value readBank;
    for (auto [i, input] : llvm::enumerate(inputs)) {
      if (scalar[i] || llvm::is_contained(
                           ArrayRef<int64_t>(groups).take_front(i), groups[i]))
        continue;
      AffineMap map = cast<AffineMapAttr>(maps[i]).getValue();
      SmallVector<Value> offsets;
      for (AffineExpr expression : map.getResults())
        offsets.push_back(
            isa<AffineDimExpr>(expression)
                ? coordinates[cast<AffineDimExpr>(expression).getPosition()]
                : index(0));
      SmallVector<OpFoldResult> at(offsets.begin(), offsets.end());
      at.back() =
          arith::AddIOp::create(b, loc, offsets.back(), index(inputOffsets[i]))
              .getResult();
      SmallVector<OpFoldResult> sizes(offsets.size(), b.getIndexAttr(1));
      SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
      int64_t inputWidth = inputWidths[i];
      int64_t elementBytes = cast<MemRefType>(input.getType())
                                 .getElementType()
                                 .getIntOrFloatBitWidth() /
                             8;
      if ((inputWidth * elementBytes + 15) / 16 > target::getTarget().bankDepth)
        return op.emitError("row_operator input row exceeds its bank");
      sizes.back() = inputWidth < 0 ? OpFoldResult(widths[i])
                                    : OpFoldResult(b.getIndexAttr(inputWidth));
      Value tile = memref::SubViewOp::create(b, loc, input, at, sizes, strides);
      Value bank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      if (inputWidth < 0 || inputWidth * elementBytes % 16)
        transfer(tile, bank, widths[i], true);
      else
        bank = BankMvinOp::create(b, loc, b.getI64Type(), tile, bank,
                                  inputWidth * elementBytes / 16, 1);
      readBank = readBank ? Value(BankTransferOp::create(b, loc, b.getI64Type(),
                                                         bank, readBank))
                          : bank;
    }
    if (!readBank)
      readBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    Value descriptor = memref::SubViewOp::create(
        b, loc, descriptors,
        ArrayRef<OpFoldResult>{varyingDescriptor ? row : index(0),
                               b.getIndexAttr(0)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(12)},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
    Value writeBank = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    writeBank =
        BankMvinOp::create(b, loc, b.getI64Type(), descriptor, writeBank, 6, 1);
    writeBank = BankRunOperatorOp::create(b, loc, b.getI64Type(), readBank,
                                          writeBank, program);
    SmallVector<OpFoldResult> at(coordinates.begin(), coordinates.end());
    SmallVector<OpFoldResult> sizes(type.getRank(), b.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(type.getRank(), b.getIndexAttr(1));
    sizes.back() = width < 0 ? OpFoldResult(outputWidth)
                             : OpFoldResult(b.getIndexAttr(width));
    Value result =
        memref::SubViewOp::create(b, loc, output, at, sizes, strides);
    if (width < 0)
      transfer(result, writeBank, outputWidth, false);
    else
      BankMvoutOp::create(b, loc, b.getI64Type(), result, writeBank, width / 4,
                          1);
    BankReleaseOp::create(b, loc, readBank);
    BankReleaseOp::create(b, loc, writeBank);
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

LogicalResult lowerPolyRowOperators(ModuleOp module) {
  SmallVector<StringAttr> operators;
  for (auto function : module.getOps<func::FuncOp>())
    if (function->hasAttr("bbacc.gp_operator"))
      operators.push_back(function.getSymNameAttr());
  if (operators.empty())
    return success();
  if (failed(lowerRowOperators(module, [&](StringAttr name) {
        auto found = llvm::find(operators, name);
        if (found == operators.end())
          llvm::report_fatal_error(
              "Poly row operator has no generated function");
        return int64_t(found - operators.begin());
      })))
    return failure();
  if (auto entry = module.lookupSymbol<func::FuncOp>("operator"))
    entry.erase();
  OpBuilder b(module.getBodyRegion());
  b.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  SmallVector<Type> arguments(8, b.getI64Type());
  auto entry = func::FuncOp::create(b, loc, "operator",
                                    b.getFunctionType(arguments, {}));
  entry->setAttr("bbacc.gp_entry", b.getUnitAttr());
  b.setInsertionPointToStart(entry.addEntryBlock());
  auto cases = llvm::to_vector(llvm::seq<int64_t>(0, operators.size()));
  auto select = scf::IndexSwitchOp::create(
      b, loc, TypeRange{},
      arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                 entry.getArgument(7)),
      cases, cases.size());
  for (auto [region, name] :
       llvm::zip_equal(select.getCaseRegions(), operators)) {
    b.setInsertionPointToStart(&region.emplaceBlock());
    func::CallOp::create(b, loc, name.getValue(), TypeRange{},
                         entry.getArguments());
    scf::YieldOp::create(b, loc);
  }
  b.setInsertionPointToStart(&select.getDefaultRegion().emplaceBlock());
  scf::YieldOp::create(b, loc);
  b.setInsertionPointAfter(select);
  func::ReturnOp::create(b, loc);
  return success();
}
