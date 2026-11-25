//===- LowerAttention.cpp - An attention in the thread's banks ------------===//
//
// A bbacc.custom "attention" (convert-linalg-to-hart's OutlineAttention.cpp)
// of q [H, Sq, D] (its rows G groups of Sg, the query heads of a key head,
// each group's rows evenly apart in memory), then `segments` parts of k^T
// [H, D, Ts] and as many of v
// [H, Ts, D] (the keys of the parts follow each other: T is the sum of the
// Ts), each the decoding of an online quantization through views, and with
// `masked` a bias [Sb, T] (FP32, Sg a multiple of Sb: query row r takes bias
// row r mod Sb), computes per head
//
//   O = dec(quant(softmax((bias + dec(quant(q)) . dec(quant(k^T))) * 2^e))) .
//   dec(quant(v))
//
// with the scores and the probabilities in banks, never in memory:
//
//   - K: the T rows of D values move into the staging bank, whole rows a DMA
//     block each, and mxint8quant packs them into the K banks, as many
//     keys a bank as it holds (packed rows of D: B of q . k^T);
//   - V: its keys go in blocks of up to bt along T (whole blocks of 32 from
//     the start of each part). A block's rows move into the staging bank as
//     K's, and mxint8quant packs its D columns along the block's keys (the
//     transpose of the staged rows) into the block's V bank: B of the term
//     of P . v over those keys;
//   - each block of mq query rows: the rows move in and mxint8quant packs
//     them; the block's bias rows move into the scores bank; mxint8mm adds
//     the block's scores against each K bank to them (or starts from +0.0
//     without a bias), its C rows T/4 bank rows apart (c_pitch) from the
//     bank's keys (c_base), so that row r of the block is the bank rows
//     [r T/4, (r + 1) T/4) of the scores bank as SoftmaxBall reads them;
//     SoftmaxBall scales by 2^e and normalizes; mxint8quant packs each V
//     block's keys of the probabilities (in_base, in_pitch T/4: A of that
//     term); mxint8mm computes the block's output in the staging bank,
//     accumulating the terms in key order, and strided DMA writes it.
//
// The quantizations are those of the graph: q and k along D, P and v along
// T, in blocks of 32 from 0 (the parts of v are whole blocks); every product
// sums its K blocks in order, a masked score from its bias (+0.0 for a key
// that is seen, so the sums are those of the graph; the softmax of a row
// gives a key that is not seen +0.0 as the graph's lowest value does). D is
// 32 or 64 (a row of q, k or v is one DMA block of at most 16 bank rows).
//
//===----------------------------------------------------------------------===//

#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mxint8mmOps.h"
#include "mxint8quantOps.h"
#include "softmaxOps.h"

using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

MemRefType bufferType(Value tensor);
SmallVector<int64_t> viewStrides(Value source, ArrayRef<Operation *> views);

LogicalResult lowerAttention(ModuleOp module) {
  SmallVector<CustomOp> calls;
  module.walk([&](CustomOp op) {
    if (op.getKind() == "attention")
      calls.push_back(op);
  });
  const auto &target = target::getTarget();
  int64_t depth = target.bankDepth;
  for (CustomOp op : calls) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    Type i64 = b.getI64Type();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    // An operand: its views back to the decoding of its quantization, then
    // the views the quantization reads its source through (nearest the
    // operand first).
    struct Input {
      SmallVector<Operation *> views;
      MXINT8ToFP32Op decode;
      FP32ToMXINT8Op quant;
      Value source;
      size_t before = 0;
    };
    int64_t segments = cast<IntegerAttr>(op->getAttr("segments")).getInt();
    bool masked = op->hasAttr("masked");
    SmallVector<Input> inputs;
    for (Value value : op.getInputs().take_front(1 + 2 * segments)) {
      Input input;
      while (Operation *def = value.getDefiningOp()) {
        if (auto decode = dyn_cast<MXINT8ToFP32Op>(def)) {
          input.decode = decode;
          input.quant = decode.getInput().getDefiningOp<FP32ToMXINT8Op>();
          break;
        }
        if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
                 tensor::ExpandShapeOp>(def))
          break;
        input.views.push_back(def);
        value = def->getOperand(0);
      }
      if (!input.quant)
        return op.emitError("attention reads quantizations of FP32 values");
      input.before = input.views.size();
      input.source = input.quant.getInput();
      while (Operation *def = input.source.getDefiningOp()) {
        if (!isa<linalg::TransposeOp, tensor::CollapseShapeOp,
                 tensor::ExpandShapeOp>(def))
          break;
        input.views.push_back(def);
        input.source = def->getOperand(0);
      }
      inputs.push_back(input);
    }
    auto shape = [&](int64_t i) {
      return cast<RankedTensorType>(op.getInputs()[i].getType()).getShape();
    };
    int64_t heads = shape(0)[0], rowsQ = shape(0)[1], d = shape(0)[2];
    // The strides of q [H, Sq, D] and each part of k [H, Ts, D] (k^T
    // transposed) and v [H, Ts, D] over their sources, D contiguous in each.
    SmallVector<SmallVector<int64_t>> strides;
    for (Input &input : inputs)
      strides.push_back(viewStrides(input.source, input.views));
    auto sourceType = [&](int64_t i) {
      return cast<RankedTensorType>(inputs[i].quant.getInput().getType());
    };
    // q and k quantized along D, v along T, each whole: the stride of the
    // quantized axis over the source, the axis that of the operand when the
    // quantization reads the source through views.
    auto quantStride = [&](int64_t i) -> int64_t {
      Input &input = inputs[i];
      if (input.views.size() == input.before)
        return bufferType(input.source)
            .getStridesAndOffset()
            .first[input.quant.getAxis()];
      // q's strides are [H, G, Sg, D] by then, its quantized axis the last.
      if (i == 0)
        return input.before == 0 ? strides[0].back() : 0;
      return input.before == 0 && strides[i].size() == 3
                 ? strides[i][input.quant.getAxis()]
                 : 0;
    };
    // The first key of each part, and whether its keys are rows in memory
    // (their D values follow each other) or columns (k^T is there: the
    // keys follow each other along each of the D values).
    SmallVector<int64_t> firstKey;
    SmallVector<bool> keyRows;
    int64_t keys = 0;
    // q as [H, G, Sg, D]: its rows evenly apart, or G groups of rows a
    // collapse merges (the query heads of a key head, in memory along
    // another axis than their tokens).
    int64_t groups = 1, perGroup = rowsQ;
    if (strides[0].size() == 3) {
      strides[0].insert(strides[0].begin() + 1, strides[0][1] * rowsQ);
    } else if (auto collapse = inputs[0].views.empty()
                                   ? tensor::CollapseShapeOp()
                                   : dyn_cast<tensor::CollapseShapeOp>(
                                         inputs[0].views.front())) {
      auto inner =
          viewStrides(inputs[0].source,
                      ArrayRef<Operation *>(inputs[0].views).drop_front());
      auto merged = collapse.getReassociationIndices();
      auto type = collapse.getSrcType();
      // The axes of a merged group that are not one.
      auto axes = [&](int64_t group) {
        SmallVector<int64_t> result;
        for (int64_t axis : merged[group])
          if (type.getDimSize(axis) != 1)
            result.push_back(axis);
        return result;
      };
      if (merged.size() == 3 && int64_t(inner.size()) == type.getRank() &&
          axes(0).size() <= 1 && axes(1).size() == 2 && merged[2].size() == 1) {
        groups = type.getDimSize(axes(1)[0]);
        perGroup = type.getDimSize(axes(1)[1]);
        strides[0] = {
            axes(0).empty() ? inner[merged[0].back()] : inner[axes(0)[0]],
            inner[axes(1)[0]], inner[axes(1)[1]], inner[merged[2][0]]};
      }
    }
    bool whole = strides[0].size() == 4 && strides[0][3] == 1 &&
                 quantStride(0) == 1 &&
                 sourceType(0).getDimSize(inputs[0].quant.getAxis()) == d;
    for (int64_t s = 0; s < segments && whole; ++s) {
      int64_t ik = 1 + s, iv = 1 + segments + s, count = shape(ik)[2];
      firstKey.push_back(keys);
      keys += count;
      if (strides[ik].size() != 3 || strides[iv].size() != 3) {
        whole = false;
        break;
      }
      // k^T's quantized axis is D, the last once its strides are [H, Ts, D].
      int64_t quantK = quantStride(ik);
      std::swap(strides[ik][1], strides[ik][2]);
      keyRows.push_back(strides[ik][2] == 1);
      whole = count % 32 == 0 && shape(iv)[1] == count &&
              (strides[ik][2] == 1 || strides[ik][1] == 1) &&
              strides[iv][2] == 1 && quantK == strides[ik][2] &&
              quantStride(iv) == strides[iv][1] &&
              sourceType(ik).getDimSize(inputs[ik].quant.getAxis()) == d &&
              sourceType(iv).getDimSize(inputs[iv].quant.getAxis()) == count;
    }
    auto packedRows = [](int64_t k) { return k / 16 + (k + 511) / 512; };
    int64_t packedD = packedRows(d);
    if (!whole || d % 32 || keys < 16 || keys > 2048 || keys / 4 > depth)
      return op.emitError("attention needs D a multiple of 32 contiguous, "
                          "parts of whole blocks of 32 keys, T up to 2048, q "
                          "and k quantized along D and v along T, whole");
    // The bias: its rows contiguous, T values each.
    Value bias;
    int64_t period = perGroup;
    if (masked) {
      Value value = op.getInputs()[1 + 2 * segments];
      auto type = cast<RankedTensorType>(value.getType());
      MemRefType buffer = bufferType(value);
      if (type.getRank() != 2 || type.getDimSize(1) != keys ||
          perGroup % type.getDimSize(0) ||
          buffer.getStridesAndOffset().first != ArrayRef<int64_t>{keys, 1})
        return op.emitError("attention bias must be contiguous [Sb, T] rows, "
                            "Sb dividing Sg");
      period = type.getDimSize(0);
      bias = bufferization::ToBufferOp::create(b, loc, buffer, value);
      bias.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
    }
    // The keys of a K bank (the last bank the rest), a multiple of 4 that
    // mxint8mm takes as its N.
    int64_t nk = std::min<int64_t>(depth / packedD, 1020) / 4 * 4;
    int64_t numK = (keys + nk - 1) / nk;
    // The keys the staging bank holds, and V's blocks of them: (part, first
    // key in the part, keys).
    int64_t staged = depth / (d / 4);
    int64_t bt = staged / 32 * 32;
    struct Block {
      int64_t part, first, count;
    };
    SmallVector<Block> vBlocks;
    int64_t packedT = 0;
    for (int64_t s = 0; s < segments; ++s)
      for (int64_t first = 0; first < shape(1 + s)[2]; first += bt) {
        int64_t count = std::min(bt, shape(1 + s)[2] - first);
        vBlocks.push_back({s, first, count});
        packedT += packedRows(count);
      }
    // The query rows of a block: their scores fill the scores bank, their
    // packed probabilities the packing bank.
    int64_t mq =
        std::min({depth / (keys / 4), depth / packedT, period, int64_t(1023)});
    // The staging bank (the output of a block too), the K and V banks, q,
    // the scores, the probabilities and their packing.
    // The keys and values stay in the banks while the queries pass, when
    // the banks hold them; else they are packed once into memory and pass
    // with every block of queries (below).
    bool resident = (d == 32 || d == 64) &&
                    !llvm::is_contained(keyRows, false) && nk >= 4 && mq >= 1 &&
                    bt >= 32 &&
                    1 + numK + int64_t(vBlocks.size()) + 4 <= target.bankNum;
    auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult(0).getUsers().begin());
    if (!destination || !op.getResult(0).hasOneUse())
      return op.emitError("attention must write its hart destination");

    // The buffers: q and each part of k and v as [H, rows, D] views of their
    // sources.
    auto view = [&](int64_t i, int64_t rows) -> Value {
      Value source = inputs[i].source;
      Value memory =
          bufferization::ToBufferOp::create(b, loc, bufferType(source), source);
      memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      Value offset =
          memref::ExtractStridedMetadataOp::create(b, loc, memory).getOffset();
      return memref::ReinterpretCastOp::create(
          b, loc,
          MemRefType::get({heads, rows, d}, b.getF32Type(),
                          StridedLayoutAttr::get(b.getContext(),
                                                 ShapedType::kDynamic,
                                                 strides[i])),
          memory, offset,
          ArrayRef<OpFoldResult>{b.getIndexAttr(heads), b.getIndexAttr(rows),
                                 b.getIndexAttr(d)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(strides[i][0]),
                                 b.getIndexAttr(strides[i][1]),
                                 b.getIndexAttr(strides[i][2])});
    };
    // q as [H, G, Sg, D].
    Value q;
    {
      Value source = inputs[0].source;
      Value memory =
          bufferization::ToBufferOp::create(b, loc, bufferType(source), source);
      memory.getDefiningOp()->setAttr("read_only", b.getUnitAttr());
      Value offset =
          memref::ExtractStridedMetadataOp::create(b, loc, memory).getOffset();
      q = memref::ReinterpretCastOp::create(
          b, loc,
          MemRefType::get({heads, groups, perGroup, d}, b.getF32Type(),
                          StridedLayoutAttr::get(b.getContext(),
                                                 ShapedType::kDynamic,
                                                 strides[0])),
          memory, offset,
          ArrayRef<OpFoldResult>{b.getIndexAttr(heads), b.getIndexAttr(groups),
                                 b.getIndexAttr(perGroup), b.getIndexAttr(d)},
          ArrayRef<OpFoldResult>{
              b.getIndexAttr(strides[0][0]), b.getIndexAttr(strides[0][1]),
              b.getIndexAttr(strides[0][2]), b.getIndexAttr(1)});
    }
    SmallVector<Value> k, v;
    for (int64_t s = 0; s < segments; ++s) {
      k.push_back(view(1 + s, shape(1 + s)[2]));
      v.push_back(view(1 + segments + s, shape(1 + s)[2]));
    }
    Value out = bufferization::ToBufferOp::create(
        b, loc, bufferType(destination.getDest()), destination.getDest());
    auto outStrides =
        cast<MemRefType>(out.getType()).getStridesAndOffset().first;
    if (outStrides[2] != 1 || outStrides[1] * 4 % 16)
      return op.emitError("attention output rows must be contiguous rows");
    // One pass (below): the blocks of query rows, each with a scores bank,
    // their packed probabilities one after another in the packing banks; the
    // keys of each K bank and V's blocks, each what the staging bank holds
    // of a piece.
    struct PassBlock {
      int64_t group, first, row, count, packedBank, packedAt;
    };
    SmallVector<PassBlock> passBlocks;
    SmallVector<SmallVector<Block>> passFills;
    SmallVector<Block> vBlocksOf;
    int64_t passPacked = 0;
    bool onePass = false;
    // Rows that follow each other in memory (q's, the keys', the values' and
    // the output's) move whole, a block of rows a DMA transfer, and D is one
    // piece; else D is in pieces of w values, as streamed (below).
    int64_t passW = d % 64 ? 32 : 64;
    if (strides[0][2] == d && outStrides[1] == d &&
        llvm::all_of(llvm::seq<int64_t>(1, 1 + 2 * segments),
                     [&](int64_t i) { return strides[i][1] == d; }))
      passW = d;
    if (!resident && !llvm::is_contained(keyRows, false)) {
      int64_t w = passW, pieces = d / w, wr = w / 4;
      int64_t pw = packedRows(w);
      int64_t held = std::min<int64_t>(depth / wr, 2048) / 32 * 32;
      int64_t chunk = std::min<int64_t>(depth / pw, 1020) / 4 * 4;
      int64_t packedKeys = 0;
      for (int64_t s = 0; s < segments; ++s)
        for (int64_t first = 0; first < shape(1 + s)[2]; first += held) {
          int64_t count = std::min(held, shape(1 + s)[2] - first);
          vBlocksOf.push_back({s, first, count});
          packedKeys += packedRows(count);
          if (passFills.empty() || count > chunk)
            passFills.emplace_back();
          int64_t inBank = 0;
          for (Block &keysOf : passFills.back())
            inBank += keysOf.count;
          if (inBank + count > chunk)
            passFills.emplace_back();
          passFills.back().push_back({s, first, count});
        }
      int64_t rowsOfBlock =
          std::min({depth / (keys / 4), period, int64_t(1023)});
      int64_t at = 0;
      for (int64_t group = 0; group < groups; ++group)
        for (int64_t base = 0; base < perGroup; base += period)
          for (int64_t first = 0; first < period; first += rowsOfBlock) {
            int64_t count = std::min(rowsOfBlock, period - first);
            if (at + count * packedKeys > depth) {
              ++passPacked;
              at = 0;
            }
            passBlocks.push_back(
                {group, base + first, first, count, passPacked, at});
            at += count * packedKeys;
          }
      ++passPacked;
      int64_t rowsAll = groups * perGroup;
      onePass = held >= 32 && chunk >= 4 && rowsOfBlock >= 1 &&
                w * packedRows(held) <= depth &&
                rowsOfBlock * packedKeys <= depth &&
                (pieces == 1 ? rowsOfBlock * wr <= depth
                             : rowsOfBlock * pieces * wr <= 1024) &&
                rowsAll * pieces * pw <= depth &&
                5 + int64_t(passBlocks.size()) + passPacked <= target.bankNum;
    }
    // Rows [first, first + count) of head `head` of `memory`.
    auto rows = [&](Value memory, Value head, OpFoldResult first,
                    int64_t count) -> Value {
      return memref::SubViewOp::create(
          b, loc, memory,
          ArrayRef<OpFoldResult>{head, first, b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count),
                                 b.getIndexAttr(d)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                 b.getIndexAttr(1)});
    };
    auto log2 = [](int64_t value) { return llvm::Log2_64(value); };
    auto alloc = [&]() -> Value {
      return BankAllocOp::create(b, loc, i64, 1, 1);
    };
    if (resident) {
      Value staging = alloc();
      // `count` rows of D values of `memory`, `stride` values apart, into the
      // staging bank from its first row: a DMA block of D/4 bank rows each.
      auto stage = [&](Value memory, int64_t count, int64_t stride) {
        staging = BankMvinStridedOp::create(
            b, loc, i64, memory, staging, index(0), index(count * d / 4),
            index(stride * 4), 0, d / 4, index(16), index(log2(d / 4)));
      };
      SmallVector<Value> kBanks, vBanks;
      for (int64_t j = 0; j < numK; ++j)
        kBanks.push_back(alloc());
      for (size_t j = 0; j < vBlocks.size(); ++j)
        vBanks.push_back(alloc());
      Value qBank = alloc(), scores = alloc(), probabilities = alloc(),
            packed = alloc();
      // A stream loop of `count` steps. A bank is the same bank in every step
      // and after the loop: after it, each holds its value from before.
      SmallVector<Value *> banks{&staging, &qBank, &scores, &probabilities,
                                 &packed};
      for (Value &bank : llvm::concat<Value>(kBanks, vBanks))
        banks.push_back(&bank);
      auto loop = [&](int64_t count, function_ref<void(Value)> body) {
        SmallVector<Value> before;
        for (Value *bank : banks)
          before.push_back(*bank);
        {
          OpBuilder::InsertionGuard guard(b);
          auto forOp =
              scf::ForOp::create(b, loc, index(0), index(count), index(1));
          forOp->setAttr("hart.stream", b.getUnitAttr());
          b.setInsertionPointToStart(forOp.getBody());
          body(forOp.getInductionVar());
        }
        for (auto [bank, value] : llvm::zip_equal(banks, before))
          *bank = value;
      };
      int64_t scale = cast<IntegerAttr>(op->getAttr("scale_log2")).getInt();
      loop(heads, [&](Value head) {
        // K: the keys of each part, as many a move as the staging bank and
        // their K bank take, packed into that bank.
        for (int64_t s = 0; s < segments; ++s)
          for (int64_t first = 0, count; first < shape(1 + s)[2];
               first += count) {
            int64_t key = firstKey[s] + first;
            count = std::min({staged, shape(1 + s)[2] - first, nk - key % nk});
            stage(rows(k[s], head, b.getIndexAttr(first), count), count,
                  strides[1 + s][1]);
            kBanks[key / nk] =
                ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                    b, loc, i64, staging, kBanks[key / nk], count, d, false, 0,
                    key % nk * packedD, d / 4, (key % nk * packedD) != 0);
          }
        // V: a block's rows, their D columns packed along its keys.
        for (auto [j, block] : llvm::enumerate(vBlocks)) {
          stage(rows(v[block.part], head, b.getIndexAttr(block.first),
                     block.count),
                block.count, strides[1 + segments + block.part][1]);
          vBanks[j] = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
              b, loc, i64, staging, vBanks[j], d, block.count, true, 0, 0,
              d / 4, false);
        }
        // A block of `m` query rows from row `first` of group `group`, `row`
        // of their bias.
        auto block = [&](int64_t group, Value first, Value row, int64_t m) {
          stage(
              memref::SubViewOp::create(
                  b, loc, q,
                  ArrayRef<OpFoldResult>{head, b.getIndexAttr(group), first,
                                         b.getIndexAttr(0)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                         b.getIndexAttr(m), b.getIndexAttr(d)},
                  ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                         b.getIndexAttr(1), b.getIndexAttr(1)}),
              m, strides[0][2]);
          qBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
              b, loc, i64, staging, qBank, m, d, false, 0, 0, d / 4, false);
          if (masked)
            scores = BankMvinOp::create(
                b, loc, i64,
                memref::SubViewOp::create(
                    b, loc, bias,
                    ArrayRef<OpFoldResult>{row, b.getIndexAttr(0)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(m),
                                           b.getIndexAttr(keys)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                           b.getIndexAttr(1)}),
                scores, m * keys / 4, 1);
          for (int64_t j = 0; j < numK; ++j)
            scores = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                b, loc, i64, qBank, kBanks[j], scores, m,
                std::min(nk, keys - j * nk), d, masked, false, 0, keys / 4,
                j * nk / 4, numK > 1 && (masked || j > 0));
          probabilities = ::buddy::softmax_ball::BankSoftmaxOp::create(
              b, loc, i64, scores, probabilities, m, keys, scale);
          // Each V block's keys of the probabilities, then its term.
          int64_t at = 0;
          for (Block &term : vBlocks) {
            packed = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, probabilities, packed, m, term.count, false,
                (firstKey[term.part] + term.first) / 4, at, keys / 4, at != 0);
            at += m * packedRows(term.count);
          }
          at = 0;
          for (auto [j, term] : llvm::enumerate(vBlocks)) {
            staging = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                b, loc, i64, packed, vBanks[j], staging, m, d, term.count,
                j > 0, false, at, d / 4, 0, false);
            at += m * packedRows(term.count);
          }
          staging = BankMvoutStridedOp::create(
              b, loc, i64,
              rows(out, head,
                   arith::AddIOp::create(b, loc, first, index(group * perGroup))
                       .getResult(),
                   m),
              staging, index(0), index(m * d / 4), index(outStrides[1] * 4), 0,
              d / 4, index(16), index(log2(d / 4)));
        };
        // The query rows of each group a bias row at a time serves, in
        // blocks.
        for (int64_t group = 0; group < groups; ++group)
          for (int64_t base = 0; base < perGroup; base += period) {
            if (period / mq)
              loop(period / mq, [&](Value at) {
                Value row = arith::MulIOp::create(b, loc, at, index(mq));
                block(group, arith::AddIOp::create(b, loc, row, index(base)),
                      row, mq);
              });
            if (period % mq)
              block(group, index(base + period / mq * mq),
                    index(period / mq * mq), period % mq);
          }
      });
      for (Value *bank : banks)
        BankReleaseOp::create(b, loc, *bank);

    } else if (onePass) {
      // One pass: every block of query rows has its own scores bank, so the
      // keys and the values pass once, packed in the banks and never
      // written to memory.
      int64_t w = passW, pieces = d / w, wr = w / 4;
      int64_t pw = packedRows(w);
      // A block's output takes its scores bank once its probabilities are
      // packed (strided DMA writes from the first 1024 bank rows).
      Value staging = alloc(), qBank = alloc(), kBank = alloc(),
            vBank = alloc(), probabilities = alloc();
      SmallVector<Value> scores, packed;
      for (size_t j = 0; j < passBlocks.size(); ++j)
        scores.push_back(alloc());
      for (int64_t j = 0; j < passPacked; ++j)
        packed.push_back(alloc());
      auto stage = [&](Value memory, int64_t count, int64_t stride) {
        staging =
            pieces == 1
                ? Value(BankMvinOp::create(b, loc, i64, memory, staging,
                                           count * wr, 1))
                : Value(BankMvinStridedOp::create(
                      b, loc, i64, memory, staging, index(0), index(count * wr),
                      index(stride * 4), 0, wr, index(16), index(log2(wr))));
      };
      auto part = [&](Value memory, Value head, OpFoldResult first,
                      int64_t count, int64_t piece) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory,
            ArrayRef<OpFoldResult>{head, first, b.getIndexAttr(piece * w)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count),
                                   b.getIndexAttr(w)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   b.getIndexAttr(1)});
      };
      SmallVector<Value> before{staging, qBank, kBank, vBank, probabilities};
      before.append(scores.begin(), scores.end());
      before.append(packed.begin(), packed.end());
      int64_t scale = cast<IntegerAttr>(op->getAttr("scale_log2")).getInt();
      {
        OpBuilder::InsertionGuard guard(b);
        auto forOp =
            scf::ForOp::create(b, loc, index(0), index(heads), index(1));
        forOp->setAttr("hart.stream", b.getUnitAttr());
        b.setInsertionPointToStart(forOp.getBody());
        Value head = forOp.getInductionVar();
        // Each block's bias rows into its scores bank and its query rows,
        // a piece at a time, packed into the q bank.
        int64_t rowsBefore = 0;
        SmallVector<int64_t> qAt;
        for (auto [j, block] : llvm::enumerate(passBlocks)) {
          qAt.push_back(rowsBefore * pieces * pw);
          if (masked)
            scores[j] = BankMvinOp::create(
                b, loc, i64,
                memref::SubViewOp::create(
                    b, loc, bias,
                    ArrayRef<OpFoldResult>{b.getIndexAttr(block.row),
                                           b.getIndexAttr(0)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(block.count),
                                           b.getIndexAttr(keys)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                           b.getIndexAttr(1)}),
                scores[j], block.count * keys / 4, 1);
          for (int64_t piece = 0; piece < pieces; ++piece) {
            stage(memref::SubViewOp::create(
                      b, loc, q,
                      ArrayRef<OpFoldResult>{head, b.getIndexAttr(block.group),
                                             b.getIndexAttr(block.first),
                                             b.getIndexAttr(piece * w)},
                      ArrayRef<OpFoldResult>{
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(block.count), b.getIndexAttr(w)},
                      ArrayRef<OpFoldResult>{
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(1), b.getIndexAttr(1)}),
                  block.count, strides[0][2]);
            qBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, staging, qBank, block.count, w, false, 0,
                qAt[j] + piece * block.count * pw, wr,
                (qAt[j] + piece * block.count * pw) != 0);
          }
          rowsBefore += block.count;
        }
        // K: the keys of a K bank, a piece at a time, staged and packed
        // into the bank; every block adds its scores against them.
        for (auto &fill : passFills) {
          int64_t firstOfFill =
              firstKey[fill.front().part] + fill.front().first;
          int64_t count = 0;
          for (Block &keysOf : fill)
            count += keysOf.count;
          for (int64_t piece = 0; piece < pieces; ++piece) {
            int64_t at = 0;
            for (Block &keysOf : fill) {
              stage(part(k[keysOf.part], head, b.getIndexAttr(keysOf.first),
                         keysOf.count, piece),
                    keysOf.count, strides[1 + keysOf.part][1]);
              kBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                  b, loc, i64, staging, kBank, keysOf.count, w, false, 0,
                  at * pw, wr, (at * pw) != 0);
              at += keysOf.count;
            }
            for (auto [j, block] : llvm::enumerate(passBlocks))
              scores[j] = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                  b, loc, i64, qBank, kBank, scores[j], block.count, count, w,
                  masked || piece > 0, false, qAt[j] + piece * block.count * pw,
                  keys / 4, firstOfFill / 4,
                  count < keys && (masked || firstOfFill > 0));
          }
        }
        // The probabilities of each block, packed along each V block's keys.
        for (auto [j, block] : llvm::enumerate(passBlocks)) {
          probabilities = ::buddy::softmax_ball::BankSoftmaxOp::create(
              b, loc, i64, scores[j], probabilities, block.count, keys, scale);
          int64_t at = block.packedAt;
          for (Block &term : vBlocksOf) {
            packed[block.packedBank] =
                ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                    b, loc, i64, probabilities, packed[block.packedBank],
                    block.count, term.count, false,
                    (firstKey[term.part] + term.first) / 4, at, keys / 4,
                    at != 0);
            at += block.count * packedRows(term.count);
          }
        }
        // V: a piece of a block of keys staged and packed along its keys;
        // every block of query rows adds its term of the output.
        int64_t termRows = 0;
        for (auto [t, term] : llvm::enumerate(vBlocksOf)) {
          for (int64_t piece = 0; piece < pieces; ++piece) {
            stage(part(v[term.part], head, b.getIndexAttr(term.first),
                       term.count, piece),
                  term.count, strides[1 + segments + term.part][1]);
            vBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, staging, vBank, w, term.count, true, 0, 0, wr,
                false);
            for (auto [j, block] : llvm::enumerate(passBlocks))
              scores[j] = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                  b, loc, i64, packed[block.packedBank], vBank, scores[j],
                  block.count, w, term.count, t > 0, false,
                  block.packedAt + block.count * termRows, wr,
                  piece * block.count * wr, pieces > 1 && (t > 0 || piece > 0));
          }
          termRows += packedRows(term.count);
        }
        for (auto [j, block] : llvm::enumerate(passBlocks))
          for (int64_t piece = 0; piece < pieces; ++piece)
            scores[j] =
                pieces == 1
                    ? Value(BankMvoutOp::create(
                          b, loc, i64,
                          part(out, head,
                               b.getIndexAttr(block.group * perGroup +
                                              block.first),
                               block.count, piece),
                          scores[j], block.count * wr, 1))
                    : Value(BankMvoutStridedOp::create(
                          b, loc, i64,
                          part(out, head,
                               b.getIndexAttr(block.group * perGroup +
                                              block.first),
                               block.count, piece),
                          scores[j], index(piece * block.count * wr),
                          index(block.count * wr), index(outStrides[1] * 4), 0,
                          wr, index(16), index(log2(wr))));
      }
      for (Value bank : before)
        BankReleaseOp::create(b, loc, bank);
    } else {
      // Streamed: D in pieces of w values (a DMA block of w/4 bank rows a
      // row), each piece of q and k packed on its own, so that a product
      // over D is the sum of its pieces' (accumulate), in the order of the
      // blocks of 32. K, a piece at a time, and V, a piece of a block of
      // keys at a time, are packed once a head into memory, in whole bank
      // rows; each block of query rows then reads them back, a bank at a
      // time.
      int64_t w = d % 64 ? 32 : 64, pieces = d / w, wr = w / 4;
      int64_t pw = packedRows(w);
      // The keys a staging bank holds, V's blocks of them and the first
      // packed bank row of each block's first piece.
      int64_t held = std::min<int64_t>(depth / wr, 2048) / 32 * 32;
      SmallVector<Block> terms;
      SmallVector<int64_t> termRow;
      int64_t packedKeys = 0, vRows = 0;
      for (int64_t s = 0; s < segments; ++s)
        for (int64_t first = 0; first < shape(1 + s)[2]; first += held) {
          int64_t count = std::min(held, shape(1 + s)[2] - first);
          terms.push_back({s, first, count});
          termRow.push_back(vRows);
          vRows += pieces * w * packedRows(count);
          packedKeys += packedRows(count);
        }
      // The keys of a K bank, a multiple of 4.
      int64_t chunk = std::min<int64_t>(depth / pw, 1020) / 4 * 4;
      int64_t m = std::min({depth / (keys / 4), depth / packedKeys,
                            depth / (pieces * wr), depth / (pieces * pw),
                            period, int64_t(1023)});
      if (held < 32 || chunk < 4 || m < 1 || w * packedRows(held) > depth)
        return op.emitError("attention exceeds the thread's banks");
      // Whole blocks of query rows when at least half as many rows a block
      // divide the rows a bias row serves: one stream loop, no last block.
      for (int64_t divisor = m; 2 * divisor > m; --divisor)
        if (period % divisor == 0) {
          m = divisor;
          break;
        }
      Type i8 = b.getI8Type();
      auto scratch = [&](int64_t bankRows) -> Value {
        auto memory = memref::AllocOp::create(
            b, loc, MemRefType::get({bankRows, 16}, i8));
        memory->setAttr("alignment", b.getI64IntegerAttr(64));
        return memory;
      };
      Value packedK = scratch(pieces * keys * pw), packedV = scratch(vRows);
      auto packedAt = [&](Value memory, OpFoldResult first,
                          int64_t count) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{first, b.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(count), b.getIndexAttr(16)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
      };
      // Columns [piece w, (piece + 1) w) of rows [first, first + count) of
      // head `head` of `memory` [H, rows, D].
      auto part = [&](Value memory, Value head, OpFoldResult first,
                      int64_t count, int64_t piece) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory,
            ArrayRef<OpFoldResult>{head, first, b.getIndexAttr(piece * w)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(count),
                                   b.getIndexAttr(w)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                   b.getIndexAttr(1)});
      };
      Value staging = alloc(), qBank = alloc(), kBank = alloc(),
            vBank = alloc(), scores = alloc(), probabilities = alloc(),
            packed = alloc(), output = alloc();
      // `count` piece rows of `memory`, `stride` values apart, into the
      // staging bank from its first row.
      auto stage = [&](Value memory, int64_t count, int64_t stride) {
        staging = BankMvinStridedOp::create(
            b, loc, i64, memory, staging, index(0), index(count * wr),
            index(stride * 4), 0, wr, index(16), index(log2(wr)));
      };
      SmallVector<Value *> banks{&staging, &qBank,         &kBank,  &vBank,
                                 &scores,  &probabilities, &packed, &output};
      auto loop = [&](int64_t count, function_ref<void(Value)> body) {
        SmallVector<Value> before;
        for (Value *bank : banks)
          before.push_back(*bank);
        {
          OpBuilder::InsertionGuard guard(b);
          auto forOp =
              scf::ForOp::create(b, loc, index(0), index(count), index(1));
          forOp->setAttr("hart.stream", b.getUnitAttr());
          b.setInsertionPointToStart(forOp.getBody());
          body(forOp.getInductionVar());
        }
        for (auto [bank, value] : llvm::zip_equal(banks, before))
          *bank = value;
      };
      int64_t scale = cast<IntegerAttr>(op->getAttr("scale_log2")).getInt();
      loop(heads, [&](Value head) {
        // K: a piece of the keys the staging bank holds, packed and written
        // to its place in memory. Keys that are columns in memory move as
        // the piece's w rows of up to 64 keys (a DMA block each), which
        // mxint8quant packs by columns.
        // `count` keys of part `s` from key `first` (a value), piece
        // `piece`.
        auto packKeys = [&](int64_t s, Value first, int64_t count,
                            int64_t piece) {
          Value memory = part(k[s], head, first, count, piece);
          if (keyRows[s]) {
            stage(memory, count, strides[1 + s][1]);
            kBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, staging, kBank, count, w, false, 0, 0, wr, false);
          } else {
            staging = BankMvinStridedOp::create(
                b, loc, i64, memory, staging, index(0), index(w * count / 4),
                index(strides[1 + s][2] * 4), 0, count / 4, index(16),
                index(log2(count / 4)));
            kBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, staging, kBank, count, w, true, 0, 0, count / 4,
                false);
          }
          Value at = arith::AddIOp::create(
              b, loc, arith::MulIOp::create(b, loc, first, index(pw)),
              index((piece * keys + firstKey[s]) * pw));
          kBank = BankMvoutOp::create(b, loc, i64,
                                      packedAt(packedK, at, count * pw), kBank,
                                      count * pw, 1);
        };
        for (int64_t s = 0; s < segments; ++s) {
          int64_t total = shape(1 + s)[2];
          int64_t count = std::min(keyRows[s] ? held : int64_t(64), total);
          for (int64_t piece = 0; piece < pieces; ++piece) {
            if (total / count)
              loop(total / count, [&](Value at) {
                packKeys(s, arith::MulIOp::create(b, loc, at, index(count)),
                         count, piece);
              });
            if (total % count)
              packKeys(s, index(total / count * count), total % count, piece);
          }
        }
        // V: a piece of a block's rows, its w columns packed along the
        // block's keys, written to its place in memory.
        auto packValues = [&](int64_t s, Value first, int64_t count, Value row,
                              int64_t piece) {
          int64_t size = w * packedRows(count);
          stage(part(v[s], head, first, count, piece), count,
                strides[1 + segments + s][1]);
          vBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
              b, loc, i64, staging, vBank, w, count, true, 0, 0, wr, false);
          vBank = BankMvoutOp::create(
              b, loc, i64,
              packedAt(packedV,
                       arith::AddIOp::create(b, loc, row, index(piece * size))
                           .getResult(),
                       size),
              vBank, size, 1);
        };
        for (size_t j = 0; j < terms.size(); ++j) {
          // The whole blocks of a part follow each other: one loop.
          Block term = terms[j];
          if (term.first != 0)
            continue;
          int64_t total = shape(1 + term.part)[2],
                  rowsOfTerm = pieces * w * packedRows(held);
          for (int64_t piece = 0; piece < pieces; ++piece) {
            if (total / held)
              loop(total / held, [&](Value at) {
                packValues(
                    term.part, arith::MulIOp::create(b, loc, at, index(held)),
                    held,
                    arith::AddIOp::create(
                        b, loc,
                        arith::MulIOp::create(b, loc, at, index(rowsOfTerm)),
                        index(termRow[j])),
                    piece);
              });
            if (total % held)
              packValues(term.part, index(total / held * held), total % held,
                         index(termRow[j + total / held]), piece);
          }
        }
        // A block of `rowsOf` query rows from row `first` of group `group`,
        // `row` of their bias.
        auto block = [&](Value group, Value first, Value row, int64_t rowsOf) {
          if (masked)
            scores = BankMvinOp::create(
                b, loc, i64,
                memref::SubViewOp::create(
                    b, loc, bias,
                    ArrayRef<OpFoldResult>{row, b.getIndexAttr(0)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(rowsOf),
                                           b.getIndexAttr(keys)},
                    ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                           b.getIndexAttr(1)}),
                scores, rowsOf * keys / 4, 1);
          // The pieces of the query rows, packed one after another.
          for (int64_t piece = 0; piece < pieces; ++piece) {
            stage(memref::SubViewOp::create(
                      b, loc, q,
                      ArrayRef<OpFoldResult>{head, group, first,
                                             b.getIndexAttr(piece * w)},
                      ArrayRef<OpFoldResult>{
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(rowsOf), b.getIndexAttr(w)},
                      ArrayRef<OpFoldResult>{
                          b.getIndexAttr(1), b.getIndexAttr(1),
                          b.getIndexAttr(1), b.getIndexAttr(1)}),
                  rowsOf, strides[0][2]);
            qBank = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, staging, qBank, rowsOf, w, false, 0,
                piece * rowsOf * pw, wr, (piece * rowsOf * pw) != 0);
          }
          // The scores against the keys of a K bank at a time, the sum of
          // the pieces' products.
          for (int64_t key = 0; key < keys; key += chunk) {
            int64_t count = std::min(chunk, keys - key);
            for (int64_t piece = 0; piece < pieces; ++piece) {
              kBank = BankMvinOp::create(
                  b, loc, i64,
                  packedAt(packedK, b.getIndexAttr((piece * keys + key) * pw),
                           count * pw),
                  kBank, count * pw, 1);
              scores = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                  b, loc, i64, qBank, kBank, scores, rowsOf, count, w,
                  masked || piece > 0, false, piece * rowsOf * pw, keys / 4,
                  key / 4, count < keys && (masked || key > 0));
            }
          }
          probabilities = ::buddy::softmax_ball::BankSoftmaxOp::create(
              b, loc, i64, scores, probabilities, rowsOf, keys, scale);
          // Each V block's keys of the probabilities, then its term, a
          // piece of the output's columns at a time.
          int64_t at = 0;
          for (Block &term : terms) {
            packed = ::buddy::mxint8quant_ball::BankQuantMxint8Op::create(
                b, loc, i64, probabilities, packed, rowsOf, term.count, false,
                (firstKey[term.part] + term.first) / 4, at, keys / 4, at != 0);
            at += rowsOf * packedRows(term.count);
          }
          at = 0;
          for (auto [j, term] : llvm::enumerate(terms)) {
            int64_t size = w * packedRows(term.count);
            for (int64_t piece = 0; piece < pieces; ++piece) {
              vBank = BankMvinOp::create(
                  b, loc, i64,
                  packedAt(packedV, b.getIndexAttr(termRow[j] + piece * size),
                           size),
                  vBank, size, 1);
              output = ::buddy::mxint8mm_ball::BankMatmulMxint8Op::create(
                  b, loc, i64, packed, vBank, output, rowsOf, w, term.count,
                  j > 0, false, at, wr, piece * rowsOf * wr,
                  pieces > 1 && (j > 0 || piece > 0));
            }
            at += rowsOf * packedRows(term.count);
          }
          Value outRow = arith::AddIOp::create(
              b, loc, first,
              arith::MulIOp::create(b, loc, group, index(perGroup)));
          for (int64_t piece = 0; piece < pieces; ++piece)
            output = BankMvoutStridedOp::create(
                b, loc, i64, part(out, head, outRow, rowsOf, piece), output,
                index(piece * rowsOf * wr), index(rowsOf * wr),
                index(outStrides[1] * 4), 0, wr, index(16), index(log2(wr)));
        };
        loop(groups, [&](Value group) {
          for (int64_t base = 0; base < perGroup; base += period) {
            if (period / m)
              loop(period / m, [&](Value at) {
                Value row = arith::MulIOp::create(b, loc, at, index(m));
                block(group, arith::AddIOp::create(b, loc, row, index(base)),
                      row, m);
              });
            if (period % m)
              block(group, index(base + period / m * m), index(period / m * m),
                    period % m);
          }
        });
      });
      for (Value *bank : banks)
        BankReleaseOp::create(b, loc, *bank);
    }

    Value written = bufferization::ToTensorOp::create(
        b, loc, cast<RankedTensorType>(op.getResult(0).getType()), out, true,
        true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
    // The operands' views, decodings and quantizations.
    for (Input &input : inputs) {
      SmallVector<Operation *> ops(input.views.begin(),
                                   input.views.begin() + input.before);
      ops.append({input.decode.getOperation(), input.quant.getOperation()});
      ops.append(input.views.begin() + input.before, input.views.end());
      for (Operation *def : ops)
        if (def->use_empty())
          def->erase();
    }
  }
  return success();
}
