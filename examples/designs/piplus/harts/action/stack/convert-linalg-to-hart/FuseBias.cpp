//===- FuseBias.cpp - A product's bias as the start of its sums -----------===//
//
// Part of convert-linalg-to-hart: the sum of a linalg.matmul from zero and a
// bias row broadcast along its rows (the linear layers' + b) becomes the
// matmul starting from the broadcast bias, linalg.broadcast of the bias as a
// vector. The product then goes to the thread with its bias: mxint8mm starts
// each element at bias[col] and adds the block terms to it, so no GP operator
// adds the bias. The FP32 sums add the same terms in another order.
//
// The sum of two products of the same A, the second from zero (the two
// MXINT8 terms of a weight, as the recipe stores every Linear), becomes the
// second product starting from the first: the thread accumulates both terms
// onto the same C tiles, and no GP operator adds them.
//
// The sum of a tensor R and a product (the residual of a block and its
// output projection), the product read through a reshape or directly,
// becomes the product starting from R: from R itself when its first term
// starts from zero, from R + the broadcast bias (a linalg.generic) when from a
// bias. The thread loads R into each C tile and accumulates onto it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

void fuseBias(ModuleOp module) {
  SmallVector<linalg::GenericOp> sums;
  module.walk([&](linalg::GenericOp op) {
    Block &body = op.getRegion().front();
    if (op.getNumDpsInputs() == 2 && op.getNumResults() == 1 &&
        op.getNumParallelLoops() == 2 && op.getNumLoops() == 2 &&
        body.getOperations().size() == 2 && isa<arith::AddFOp>(body.front()) &&
        op.getIndexingMapsArray().back().isIdentity())
      sums.push_back(op);
  });
  for (linalg::GenericOp sum : sums) {
    auto maps = sum.getIndexingMapsArray();
    for (int64_t side = 0; side < 2; ++side) {
      auto product = sum.getDpsInputs()[side].getDefiningOp<linalg::MatmulOp>();
      Value bias = sum.getDpsInputs()[1 - side];
      auto biasType = cast<RankedTensorType>(bias.getType());
      auto fill = product
                      ? product.getOutputs()[0].getDefiningOp<linalg::FillOp>()
                      : linalg::FillOp();
      // The bias: [1, n] read at (0, d1).
      if (!product || !product->hasOneUse() || !maps[side].isIdentity() ||
          !fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()) ||
          biasType.getRank() != 2 || biasType.getDimSize(0) != 1 ||
          maps[1 - side] !=
              AffineMap::get(2, 0,
                             {getAffineConstantExpr(0, sum.getContext()),
                              getAffineDimExpr(1, sum.getContext())},
                             sum.getContext()))
        continue;
      OpBuilder b(sum);
      Location loc = sum.getLoc();
      Value row = tensor::CollapseShapeOp::create(
          b, loc, bias, ArrayRef<ReassociationIndices>{{0, 1}});
      Value start =
          linalg::BroadcastOp::create(b, loc, row, product.getOutputs()[0],
                                      ArrayRef<int64_t>{0})
              ->getResult(0);
      Operation *started = b.clone(*product);
      started->setOperand(2, start);
      sum.getResult(0).replaceAllUsesWith(started->getResult(0));
      sum.erase();
      product.erase();
      break;
    }
  }
  sums.clear();
  module.walk([&](linalg::GenericOp op) {
    Block &body = op.getRegion().front();
    if (op.getNumDpsInputs() == 2 && op.getNumResults() == 1 &&
        op.getNumLoops() == op.getNumParallelLoops() &&
        body.getOperations().size() == 2 && isa<arith::AddFOp>(body.front()) &&
        llvm::all_of(op.getIndexingMapsArray(),
                     [](AffineMap map) { return map.isIdentity(); }))
      sums.push_back(op);
  });
  for (linalg::GenericOp sum : sums) {
    for (int64_t side = 0; side < 2; ++side) {
      auto second = sum.getDpsInputs()[side].getDefiningOp<linalg::MatmulOp>();
      auto first =
          sum.getDpsInputs()[1 - side].getDefiningOp<linalg::MatmulOp>();
      auto fill = second
                      ? second.getOutputs()[0].getDefiningOp<linalg::FillOp>()
                      : linalg::FillOp();
      if (!second || !first || !second->hasOneUse() || !first->hasOneUse() ||
          first.getInputs()[0] != second.getInputs()[0] || !fill ||
          !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
        continue;
      OpBuilder b(sum);
      Operation *started = b.clone(*second);
      started->setOperand(2, first.getResult(0));
      sum.getResult(0).replaceAllUsesWith(started->getResult(0));
      sum.erase();
      second.erase();
      break;
    }
  }
  sums.clear();
  module.walk([&](linalg::GenericOp op) {
    Block &body = op.getRegion().front();
    if (op.getNumDpsInputs() == 2 && op.getNumResults() == 1 &&
        op.getNumLoops() == op.getNumParallelLoops() &&
        body.getOperations().size() == 2 && isa<arith::AddFOp>(body.front()) &&
        llvm::all_of(op.getIndexingMapsArray(),
                     [](AffineMap map) { return map.isIdentity(); }))
      sums.push_back(op);
  });
  for (linalg::GenericOp sum : sums) {
    for (int64_t side = 0; side < 2; ++side) {
      Value read = sum.getDpsInputs()[side];
      Value residual = sum.getDpsInputs()[1 - side];
      auto expand = read.getDefiningOp<tensor::ExpandShapeOp>();
      auto product =
          (expand ? expand.getSrc() : read).getDefiningOp<linalg::MatmulOp>();
      if (!product || !product->hasOneUse() || (expand && !expand->hasOneUse()))
        continue;
      linalg::MatmulOp head = product;
      while (auto previous =
                 head.getOutputs()[0].getDefiningOp<linalg::MatmulOp>())
        head = previous;
      Value init = head.getOutputs()[0];
      auto fill = init.getDefiningOp<linalg::FillOp>();
      auto start = init.getDefiningOp<linalg::BroadcastOp>();
      Operation *def = residual.getDefiningOp();
      if (!((fill && matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) ||
            start) ||
          (def && (def->getBlock() != head->getBlock() ||
                   !def->isBeforeInBlock(head))))
        continue;
      OpBuilder b(head);
      Location loc = sum.getLoc();
      Value rows =
          expand ? Value(tensor::CollapseShapeOp::create(
                       b, loc, residual, expand.getReassociationIndices()))
                 : residual;
      Value started = rows;
      if (start) {
        int64_t rank = cast<RankedTensorType>(init.getType()).getRank();
        auto identity = b.getMultiDimIdentityMap(rank);
        started =
            linalg::GenericOp::create(
                b, loc, init.getType(), ValueRange{rows, init},
                ValueRange{start.getInit()},
                SmallVector<AffineMap>(3, identity),
                SmallVector<utils::IteratorType>(rank,
                                                 utils::IteratorType::parallel),
                [](OpBuilder &nested, Location at, ValueRange args) {
                  linalg::YieldOp::create(nested, at,
                                          ValueRange{arith::AddFOp::create(
                                              nested, at, args[0], args[1])});
                })
                ->getResult(0);
      }
      head->setOperand(2, started);
      sum.getResult(0).replaceAllUsesWith(read);
      sum.erase();
      break;
    }
  }
}
