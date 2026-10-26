#include "Utils/StageTrace.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

LogicalResult mlir::buddy::emitStageTensorTrace(OpBuilder &builder,
                                                Location loc, Value value,
                                                DictionaryAttr metadata) {
  auto path = cast<ArrayAttr>(metadata.get("id_path"));
  if (path.size() > 4)
    return emitError(loc, "stage trace id_path exceeds runtime depth limit");
  auto type = cast<MemRefType>(value.getType());
  auto element = type.getElementType();
  auto argumentType = MemRefType::get({ShapedType::kDynamic}, element);
  constexpr StringLiteral name = "buckyballTraceStageI8Path";
  Operation *parent = builder.getInsertionBlock()->getParentOp();
  auto module = parent->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    Type i64 = builder.getI64Type();
    auto function = func::FuncOp::create(
        builder, loc, name,
        builder.getFunctionType({i64, i64, i64, i64, i64, i64, argumentType},
                                {}));
    function.setPrivate();
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  }
  Value packed;
  if (!type.getLayout().isIdentity()) {
    packed = memref::AllocOp::create(builder, loc,
                                     MemRefType::get(type.getShape(), element));
    memref::CopyOp::create(builder, loc, value, packed);
    value = packed;
  }
  int64_t size = type.getNumElements();
  if (type.getRank() != 1) {
    SmallVector<ReassociationIndices> groups(1);
    for (int64_t axis = 0; axis < type.getRank(); ++axis)
      groups[0].push_back(axis);
    value = memref::CollapseShapeOp::create(
        builder, loc, MemRefType::get({size}, element), value, groups);
  }
  value = memref::CastOp::create(builder, loc, argumentType, value);
  SmallVector<Value> arguments;
  arguments.push_back(arith::ConstantIntOp::create(
      builder, loc, cast<IntegerAttr>(metadata.get("id")).getInt(), 64));
  arguments.push_back(
      arith::ConstantIntOp::create(builder, loc, path.size(), 64));
  for (size_t index = 0; index < 4; ++index) {
    int64_t component =
        index < path.size() ? cast<IntegerAttr>(path[index]).getInt() : -1;
    arguments.push_back(
        arith::ConstantIntOp::create(builder, loc, component, 64));
  }
  arguments.push_back(value);
  func::CallOp::create(builder, loc, name, TypeRange{}, arguments);
  if (packed)
    memref::DeallocOp::create(builder, loc, packed);
  return success();
}
