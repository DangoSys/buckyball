#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "f32addOps.h"
#include "intnormOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mxint8actOps.h"
#include "mxint8quantizeOps.h"
#include "mxint8unpackOps.h"

using namespace mlir;
using namespace ::buddy::acc;
LogicalResult lowerReduceQuantizedRows(ModuleOp module);
LogicalResult lowerReduceIntegerNorm(ModuleOp module);
LogicalResult lowerReducePackedDecode(ModuleOp module);
LogicalResult outlineOperators(ModuleOp module);
LogicalResult lowerPolyRowOperators(ModuleOp module);
void prepareReduceActivations(ModuleOp module);
LogicalResult lowerReduceActivations(ModuleOp module);

namespace {
linalg::GenericOp addition(Value value) {
  auto op = value.getDefiningOp<linalg::GenericOp>();
  if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      op.getNumResults() != 1 || op.getNumParallelLoops() != op.getNumLoops())
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.getElementType().isF32() || !type.getRank() ||
      type.isDynamicDim(type.getRank() - 1) || type.getShape().back() <= 0 ||
      type.getShape().back() % 4)
    return {};
  for (AffineMap map : op.getIndexingMapsArray())
    if (!map.isIdentity())
      return {};
  Block &body = op.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2)
    return {};
  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) ||
      yield.getValues()[0] != add.getResult())
    return {};
  return op;
}

class ConvertLinalgToAccPass
    : public PassWrapper<ConvertLinalgToAccPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToAccPass)
  StringRef getArgument() const final { return "convert-linalg-to-acc"; }
  StringRef getDescription() const final {
    return "Map ordered FP32 sums and MXINT8 quantization to the reduce "
           "accelerator";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        ::buddy::mxint8act_ball::Mxint8actBallDialect, ::buddy::acc::AccDialect,
        cf::ControlFlowDialect, ::buddy::frontend::FrontendDialect,
        ::buddy::mxint8quantize_ball::Mxint8quantizeBallDialect,
        ::buddy::mxint8unpack_ball::Mxint8UnpackBallDialect,
        ::buddy::f32add_ball::F32AddBallDialect,
        ::buddy::intnorm_ball::IntnormBallDialect, arith::ArithDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
        math::MathDialect, vector::VectorDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    prepareReduceActivations(getOperation());
    if (failed(lowerReduceIntegerNorm(getOperation())))
      return signalPassFailure();
    if (failed(lowerReduceQuantizedRows(getOperation())) ||
        failed(lowerReduceActivations(getOperation())))
      return signalPassFailure();
    const auto &target = target::getTarget();
    if (target.bankWidthBits != 128 || target.bankNum < 3) {
      getOperation().emitError("reduce needs 128-bit banks and three banks");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> roots;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op.getNumResults() != 1 || !addition(op.getResult(0)))
        return;
      if (op.getResult(0).hasOneUse()) {
        auto user =
            dyn_cast<linalg::GenericOp>(*op.getResult(0).getUsers().begin());
        if (user && user.getNumResults() == 1 && addition(user.getResult(0)) &&
            user.getInputs()[0] == op.getResult(0))
          return;
      }
      roots.push_back(op);
    });
    for (auto root : roots) {
      SmallVector<linalg::GenericOp> chain;
      SmallVector<Value> terms;
      auto current = root;
      for (;;) {
        chain.push_back(current);
        terms.push_back(current.getInputs()[1]);
        Value left = current.getInputs()[0];
        auto prior = addition(left);
        if (!prior || !left.hasOneUse()) {
          terms.push_back(left);
          break;
        }
        current = prior;
      }
      std::reverse(terms.begin(), terms.end());
      OpBuilder b(root);
      Location loc = root.getLoc();
      auto type = cast<RankedTensorType>(root.getResult(0).getType());
      auto index = [&](int64_t value) {
        return arith::ConstantIndexOp::create(b, loc, value).getResult();
      };
      int64_t width = type.getShape().back();
      SmallVector<::buddy::frontend::MXINT8ToFP32Op> packedTerms;
      bool packedSum =
          width % 32 == 0 && width / 4 <= target.bankDepth && width / 32 <= 128;
      for (Value term : terms) {
        auto decode = term.getDefiningOp<::buddy::frontend::MXINT8ToFP32Op>();
        if (!packedSum || !decode || decode.getResult().getType() != type ||
            int64_t(decode.getAxis()) != type.getRank() - 1 ||
            decode.getInput().getType().getShape().drop_back() !=
                type.getShape().drop_back() ||
            decode.getInput().getType().getShape().back() !=
                width + width / 32) {
          packedSum = false;
          break;
        }
        unsigned dataUses = 0;
        for (Operation *user : term.getUsers()) {
          if (auto dim = dyn_cast<tensor::DimOp>(user)) {
            if (!dim.getConstantIndex())
              packedSum = false;
          } else if (llvm::any_of(chain, [&](linalg::GenericOp sum) {
                       return sum.getOperation() == user;
                     })) {
            ++dataUses;
          } else {
            packedSum = false;
          }
        }
        packedSum &= dataUses == 1;
        packedTerms.push_back(decode);
      }
      SmallVector<Value> dynamic;
      Value elements = index(1);
      for (int64_t axis = 0; axis < type.getRank(); ++axis) {
        Value size = type.isDynamicDim(axis)
                         ? tensor::DimOp::create(
                               b, loc,
                               packedSum ? packedTerms.front().getInput()
                                         : terms.front(),
                               axis)
                               .getResult()
                         : index(type.getDimSize(axis));
        if (type.isDynamicDim(axis))
          dynamic.push_back(size);
        elements = b.createOrFold<arith::MulIOp>(loc, elements, size);
      }
      auto memoryType = MemRefType::get(type.getShape(), b.getF32Type());
      Value output = memref::AllocOp::create(b, loc, memoryType, dynamic);
      output.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
      ReassociationIndices all;
      for (int64_t axis = 0; axis < type.getRank(); ++axis)
        all.push_back(axis);
      auto flatten = [&](Value memory) -> Value {
        auto shape = cast<MemRefType>(memory.getType());
        if (shape.getRank() == 1)
          return memory;
        auto flat =
            MemRefType::get({shape.hasStaticShape() ? shape.getNumElements()
                                                    : ShapedType::kDynamic},
                            shape.getElementType());
        return memref::CollapseShapeOp::create(
            b, loc, flat, memory, ArrayRef<ReassociationIndices>{all});
      };
      if (packedSum) {
        int64_t packed = width + width / 32;
        SmallVector<Value> inputs;
        for (auto decode : packedTerms) {
          auto inputType = decode.getInput().getType();
          Value memory = bufferization::ToBufferOp::create(
              b, loc, MemRefType::get(inputType.getShape(), b.getI8Type()),
              decode.getInput(), true);
          inputs.push_back(flatten(memory));
        }
        Value flatOutput = flatten(output);
        Value rows =
            b.createOrFold<arith::DivUIOp>(loc, elements, index(width));
        auto loop = scf::ForOp::create(b, loc, index(0), rows, index(1));
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(loop.getBody());
          Value row = loop.getInductionVar();
          Value inputOffset = arith::MulIOp::create(b, loc, row, index(packed));
          Value outputOffset = arith::MulIOp::create(b, loc, row, index(width));
          auto window = [&](Value memory, Value first, int64_t count) -> Value {
            return memref::SubViewOp::create(
                b, loc, memory, ArrayRef<OpFoldResult>{first},
                ArrayRef<OpFoldResult>{b.getIndexAttr(count)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
          };
          auto unpack = [&](Value input) -> Value {
            Value source = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            int64_t whole = packed / 16 * 16;
            source = BankMvinOp::create(b, loc, b.getI64Type(),
                                        window(input, inputOffset, whole),
                                        source, whole / 16, 1);
            if (packed % 16) {
              Value first =
                  arith::AddIOp::create(b, loc, inputOffset, index(whole));
              source = BankMvinStridedOp::create(
                  b, loc, b.getI64Type(), window(input, first, packed % 16),
                  source, index(whole / 16), index(1), index(packed % 16), 0, 1,
                  index(packed % 16), index(0));
            }
            Value blocks = arith::ConstantIntOp::create(b, loc, width / 32, 64);
            result = ::buddy::mxint8unpack_ball::BankUnpackOp::create(
                b, loc, b.getI64Type(), source, result, blocks);
            BankReleaseOp::create(b, loc, source);
            return result;
          };
          Value sum = unpack(inputs.front());
          for (Value input : ArrayRef<Value>(inputs).drop_front()) {
            Value addend = unpack(input);
            Value destination =
                BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            Value next = ::buddy::f32add_ball::BankAddF32Op::create(
                b, loc, b.getI64Type(), sum, addend, destination, width / 4);
            BankReleaseOp::create(b, loc, sum);
            BankReleaseOp::create(b, loc, addend);
            sum = next;
          }
          Value stored = BankMvoutOp::create(
              b, loc, b.getI64Type(), window(flatOutput, outputOffset, width),
              sum, width / 4, 1);
          BankReleaseOp::create(b, loc, stored);
        }
      } else {
        SmallVector<Value> inputs;
        for (Value term : terms)
          inputs.push_back(flatten(bufferization::ToBufferOp::create(
              b, loc, memoryType, term, true)));
        Value flatOutput = flatten(output);
        Value remaining =
            b.createOrFold<arith::DivUIOp>(loc, elements, index(4));
        Value offset = index(0);
        for (int64_t rows = target.bankDepth; rows; rows /= 2) {
          Value trips =
              b.createOrFold<arith::DivUIOp>(loc, remaining, index(rows));
          auto loop = scf::ForOp::create(b, loc, index(0), trips, index(1));
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(loop.getBody());
            Value first = b.createOrFold<arith::AddIOp>(
                loc, offset,
                b.createOrFold<arith::MulIOp>(loc, loop.getInductionVar(),
                                              index(rows)));
            first = b.createOrFold<arith::MulIOp>(loc, first, index(4));
            auto window = [&](Value memory) -> Value {
              return memref::SubViewOp::create(
                  b, loc, memory, ArrayRef<OpFoldResult>{first},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(rows * 4)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            };
            Value sum = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
            sum = BankMvinOp::create(b, loc, b.getI64Type(), window(inputs[0]),
                                     sum, rows, 1);
            for (Value input : ArrayRef<Value>(inputs).drop_front()) {
              Value addend = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              addend = BankMvinOp::create(b, loc, b.getI64Type(), window(input),
                                          addend, rows, 1);
              Value destination =
                  BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
              Value next = ::buddy::f32add_ball::BankAddF32Op::create(
                  b, loc, b.getI64Type(), sum, addend, destination, rows);
              BankReleaseOp::create(b, loc, sum);
              BankReleaseOp::create(b, loc, addend);
              sum = next;
            }
            Value stored = BankMvoutOp::create(
                b, loc, b.getI64Type(), window(flatOutput), sum, rows, 1);
            BankReleaseOp::create(b, loc, stored);
          }
          offset = b.createOrFold<arith::AddIOp>(
              loc, offset,
              b.createOrFold<arith::MulIOp>(loc, trips, index(rows)));
          remaining =
              b.createOrFold<arith::RemUIOp>(loc, remaining, index(rows));
        }
      }
      Value result =
          bufferization::ToTensorOp::create(b, loc, type, output, true, true);
      root.getResult(0).replaceAllUsesWith(result);
      if (packedSum)
        for (auto decode : packedTerms)
          for (Operation *user :
               llvm::make_early_inc_range(decode.getResult().getUsers()))
            if (auto dim = dyn_cast<tensor::DimOp>(user)) {
              OpBuilder shape(dim);
              int64_t axis = *dim.getConstantIndex();
              Value size =
                  axis == type.getRank() - 1
                      ? Value(arith::ConstantIndexOp::create(
                            shape, dim.getLoc(), width))
                      : Value(tensor::DimOp::create(shape, dim.getLoc(),
                                                    decode.getInput(), axis));
              dim.getResult().replaceAllUsesWith(size);
              dim.erase();
            }
      for (auto op : chain)
        op.erase();
      if (packedSum)
        for (auto decode : packedTerms)
          decode.erase();
    }
    if (failed(lowerReducePackedDecode(getOperation())) ||
        failed(outlineOperators(getOperation())))
      return signalPassFailure();
    SmallVector<CustomOp> operators;
    getOperation().walk([&](CustomOp op) {
      if (op.getKind() == "row_operator")
        operators.push_back(op);
    });
    for (CustomOp op : operators) {
      if (op.getResult(0).hasOneUse() &&
          isa<bufferization::MaterializeInDestinationOp>(
              *op.getResult(0).getUsers().begin()))
        continue;
      OpBuilder b(op);
      auto type = cast<RankedTensorType>(op.getResult(0).getType());
      auto maps = op->getAttrOfType<ArrayAttr>("maps");
      SmallVector<Value> dimensions;
      for (int64_t axis = 0; axis < type.getRank(); ++axis) {
        if (!type.isDynamicDim(axis))
          continue;
        Value extent;
        for (auto [input, map] :
             llvm::zip_equal(op.getInputs(), maps.getValue().drop_back()))
          for (auto [index, expression] : llvm::enumerate(
                   cast<AffineMapAttr>(map).getValue().getResults()))
            if (auto dim = dyn_cast<AffineDimExpr>(expression);
                dim && dim.getPosition() == axis && !extent)
              extent = tensor::DimOp::create(b, op.getLoc(), input, index);
        if (!extent) {
          op.emitError("GP output dimension has no input extent");
          return signalPassFailure();
        }
        dimensions.push_back(extent);
      }
      Value output = tensor::EmptyOp::create(b, op.getLoc(), type.getShape(),
                                             type.getElementType(), dimensions);
      b.setInsertionPointAfter(op);
      auto written = bufferization::MaterializeInDestinationOp::create(
          b, op.getLoc(), op.getResult(0), output);
      op.getResult(0).replaceAllUsesExcept(written.getResult(), written);
    }
    // Dedicated output arguments need no temporary allocation or final copy.
    getOperation().walk([&](bufferization::MaterializeInDestinationOp op) {
      auto collapse = op.getSource().getDefiningOp<tensor::CollapseShapeOp>();
      Value source = collapse ? collapse.getSrc() : op.getSource();
      auto tensor = source.getDefiningOp<bufferization::ToTensorOp>();
      auto destination = dyn_cast<BlockArgument>(op.getDest());
      if (!tensor || !destination ||
          !isa<RankedTensorType>(destination.getType()) ||
          !destination.hasOneUse() || destination.getOwner() != op->getBlock())
        return;
      auto allocation = tensor.getBuffer().getDefiningOp<memref::AllocOp>();
      if (!allocation || allocation->getBlock() != op->getBlock())
        return;
      OpBuilder b(allocation);
      auto type = cast<RankedTensorType>(destination.getType());
      Value output = bufferization::ToBufferOp::create(
          b, op.getLoc(),
          MemRefType::get(type.getShape(), type.getElementType()), destination);
      if (collapse) {
        SmallVector<OpFoldResult> sizes;
        auto dynamic = allocation.getDynamicSizes().begin();
        for (int64_t extent : allocation.getType().getShape())
          sizes.push_back(ShapedType::isDynamic(extent)
                              ? OpFoldResult(*dynamic++)
                              : OpFoldResult(b.getIndexAttr(extent)));
        output = memref::ExpandShapeOp::create(
            b, op.getLoc(), allocation.getType(), output,
            collapse.getReassociationIndices(), sizes);
      }
      allocation.replaceAllUsesWith(output);
      allocation.erase();
    });
    if (failed(lowerPolyRowOperators(getOperation())))
      signalPassFailure();
  }
};
PassRegistration<ConvertLinalgToAccPass> registration;
} // namespace
