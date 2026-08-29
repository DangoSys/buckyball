//===- ReportCycleEstimatePass.cpp - Static cycle estimation -------------===//
//
// Per-op latency (bemu formulas) + bank RAW + ball unit pipeline.
// Each ball unit runs one op at a time; different units overlap.
// DMA (mvin/mvout) overlaps with ball ops. Fence drains all.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
constexpr int64_t kTile = 16;
constexpr int64_t kBankWidthBits = 128;
constexpr int64_t kBankDepth = 1024;

static std::optional<int64_t> getConstI64(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ai = dyn_cast<IntegerAttr>(c.getValue()))
      return ai.getInt();
  return std::nullopt;
}

// Ball unit IDs for pipeline tracking
enum Unit {
  U_NONE,
  U_SMATMUL,
  U_MATADD,
  U_FP2INT,
  U_INT2FP,
  U_RELU,
  U_TRANS,
  U_MVIN,
  U_MVOUT,
  U_FENCE
};

class ReportCycleEstimatePass
    : public PassWrapper<ReportCycleEstimatePass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReportCycleEstimatePass)
  ReportCycleEstimatePass() = default;
  ReportCycleEstimatePass(const ReportCycleEstimatePass &) {}
  StringRef getArgument() const final { return "report-cycle-estimate"; }
  StringRef getDescription() const final {
    return "Cycle estimate with bank RAW + pipeline.";
  }
  Option<bool> verbose{*this, "verbose", llvm::cl::init(false)};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseMap<int64_t, int64_t> bankBusy;
    DenseMap<int, int64_t> unitBusy;
    int64_t total = 0, totalStall = 0, fenceStall = 0, idx = 0;
    DenseMap<StringRef, int64_t> byType, byCount;
    DenseMap<int, int64_t> unitStall, unitLat, unitOps;

    auto bFree = [&](int64_t id) -> int64_t {
      auto it = bankBusy.find(id);
      return it == bankBusy.end() ? 0 : it->second;
    };
    auto uFree = [&](int u) -> int64_t {
      auto it = unitBusy.find(u);
      return it == unitBusy.end() ? 0 : it->second;
    };

    auto emit = [&](const char *name, int64_t lat, SmallVector<int64_t> rd,
                    SmallVector<int64_t> wr, int unit, bool fence) {
      int64_t start = 0;
      int64_t stall = 0;
      if (fence) {
        for (auto &kv : bankBusy)
          start = std::max(start, kv.second);
        for (auto &kv : unitBusy)
          start = std::max(start, kv.second);
        if (start > 0) {
          fenceStall += start;
          totalStall += start;
        }
      } else {
        int64_t bankReady = 0;
        for (int64_t b : rd)
          bankReady = std::max(bankReady, bFree(b));
        for (int64_t b : wr)
          bankReady = std::max(bankReady, bFree(b));
        int64_t unitReady = (unit != U_NONE) ? uFree(unit) : 0;
        start = std::max(bankReady, unitReady);
        stall = std::max(int64_t(0), bankReady - unitReady);
        if (stall > 0)
          totalStall += stall;
        if (unit != U_NONE) {
          unitStall[unit] += stall;
          unitLat[unit] += lat;
          unitOps[unit] += 1;
        }
      }
      int64_t finish = start + lat;
      if (finish > total)
        total = finish;
      for (int64_t b : wr)
        bankBusy[b] = finish;
      if (!fence && unit != U_NONE)
        unitBusy[unit] = finish;
      byType[name] += lat;
      byCount[name] += 1;
      idx++;
      if (verbose) {
        llvm::errs() << "  [" << idx << "] " << name << " lat=" << lat
                     << " start=" << start << " finish=" << finish;
        if (stall > 0)
          llvm::errs() << " stall=" << stall;
        llvm::errs() << "\n";
      }
    };

    auto walk = [&](Block &blk, auto &self) -> void {
      for (Operation &op : blk.getOperations()) {
        if (dyn_cast<MsetOp>(op)) {
          emit("mset", 1, {}, {}, U_NONE, false);
          continue;
        }
        if (auto o = dyn_cast<MvinOp>(op)) {
          auto a = getConstI64(o.getAddr()), d = getConstI64(o.getDepth());
          emit("mvin", d ? std::max(*d, int64_t(1)) : 0, {},
               a ? SmallVector<int64_t>{*a} : SmallVector<int64_t>{}, U_MVIN,
               false);
          continue;
        }
        if (auto o = dyn_cast<MvoutOp>(op)) {
          auto a = getConstI64(o.getAddr()), d = getConstI64(o.getDepth());
          emit("mvout", d ? std::max(*d, int64_t(1)) : 0,
               a ? SmallVector<int64_t>{*a} : SmallVector<int64_t>{}, {},
               U_MVOUT, false);
          continue;
        }
        if (auto o = dyn_cast<SMatMulOp>(op)) {
          auto a = getConstI64(o.getOp1BankId()),
               b = getConstI64(o.getOp2BankId()),
               c = getConstI64(o.getResultBankId()),
               cfg = getConstI64(o.getConfig());
          if (!cfg) {
            emit("smatmul", 0, {}, {}, U_SMATMUL, false);
            continue;
          }
          int64_t r = *cfg & 0xfff, col = (*cfg >> 12) & 0xfff,
                  k = (*cfg >> 24) & 0xfff;
          SmallVector<int64_t> rd, wr;
          if (a)
            rd.push_back(*a);
          if (b)
            rd.push_back(*b);
          if (c)
            wr.push_back(*c);
          emit(o.getWs() ? "smatmul_ws" : "smatmul_os",
               r * col * k / kTile + r * col / 2, rd, wr, U_SMATMUL, false);
          continue;
        }
        if (auto o = dyn_cast<MatAddOp>(op)) {
          auto a = getConstI64(o.getABankId()), b = getConstI64(o.getBBankId()),
               c = getConstI64(o.getCBankId()), it = getConstI64(o.getIter());
          SmallVector<int64_t> rd, wr;
          if (a)
            rd.push_back(*a);
          if (b)
            rd.push_back(*b);
          if (c)
            wr.push_back(*c);
          emit("matadd", it ? *it * 4 : 0, rd, wr, U_MATADD, false);
          continue;
        }
        if (auto o = dyn_cast<Fp2IntOp>(op)) {
          auto in = getConstI64(o.getInputBankId()),
               out = getConstI64(o.getOutputBankId()),
               it = getConstI64(o.getIter());
          emit("fp2int", it ? std::max(*it, int64_t(1)) : 0,
               in ? SmallVector<int64_t>{*in} : SmallVector<int64_t>{},
               out ? SmallVector<int64_t>{*out} : SmallVector<int64_t>{},
               U_FP2INT, false);
          continue;
        }
        if (auto o = dyn_cast<Int2FpTensorOp>(op)) {
          auto in = getConstI64(o.getInputBankId()),
               out = getConstI64(o.getOutputBankId()),
               it = getConstI64(o.getIter());
          emit("int2fp", it ? std::max(*it, int64_t(1)) : 0,
               in ? SmallVector<int64_t>{*in} : SmallVector<int64_t>{},
               out ? SmallVector<int64_t>{*out} : SmallVector<int64_t>{},
               U_INT2FP, false);
          continue;
        }
        if (auto o = dyn_cast<Int2FpChannelOp>(op)) {
          auto in = getConstI64(o.getInputBankId()),
               out = getConstI64(o.getOutputBankId()),
               it = getConstI64(o.getIter());
          emit("int2fp_ch", it ? std::max(*it, int64_t(1)) : 0,
               in ? SmallVector<int64_t>{*in} : SmallVector<int64_t>{},
               out ? SmallVector<int64_t>{*out} : SmallVector<int64_t>{},
               U_INT2FP, false);
          continue;
        }
        if (auto o = dyn_cast<ReluOp>(op)) {
          auto b = getConstI64(o.getBankId()), it = getConstI64(o.getIter()),
               st = getConstI64(o.getStride());
          int64_t lat = (it && st && *st > 0) ? kBankDepth / *st * *it * 4 : 0;
          emit("relu", lat,
               b ? SmallVector<int64_t>{*b} : SmallVector<int64_t>{},
               b ? SmallVector<int64_t>{*b} : SmallVector<int64_t>{}, U_RELU,
               false);
          continue;
        }
        if (auto o = dyn_cast<TransposeOp>(op)) {
          auto src = getConstI64(o.getInputBankId()),
               dst = getConstI64(o.getOutputBankId()),
               it = getConstI64(o.getIter());
          int64_t epg = kBankWidthBits / 8;
          emit("transpose", it ? *it * epg * 2 : 0,
               src ? SmallVector<int64_t>{*src} : SmallVector<int64_t>{},
               dst ? SmallVector<int64_t>{*dst} : SmallVector<int64_t>{},
               U_TRANS, false);
          continue;
        }
        if (dyn_cast<FenceOp>(op)) {
          emit("fence", 1, {}, {}, U_FENCE, true);
          continue;
        }
        if (auto f = dyn_cast<scf::ForOp>(op)) {
          self(*f.getBody(), self);
          continue;
        }
        if (auto f = dyn_cast<scf::IfOp>(op)) {
          self(*f.thenBlock(), self);
          if (auto *e = f.elseBlock())
            self(*e, self);
          continue;
        }
      }
    };

    for (Block &blk : func.getBlocks())
      walk(blk, walk);

    llvm::errs() << "\n[cycle-estimate] " << func.getName() << "\n";
    llvm::errs() << "  total:  " << total << "\n";
    llvm::errs() << "  stall:  " << totalStall << " (fence: " << fenceStall
                 << ")\n";
    if (total > 0)
      llvm::errs() << "  stall%: " << 100 * totalStall / total << "\n";
    for (auto &kv : byType)
      llvm::errs() << "  " << kv.first << ": " << kv.second << " ("
                   << byCount[kv.first] << " ops)\n";

    // Per-unit stall breakdown
    static const char *uNames[] = {"none",   "smatmul", "matadd",    "fp2int",
                                   "int2fp", "relu",    "transpose", "mvin",
                                   "mvout",  "fence"};
    llvm::errs() << "  --- per-unit (blocked = unit free but bank busy) ---\n";
    for (int u = U_SMATMUL; u <= U_MVOUT; ++u) {
      if (unitOps.count(u) && unitOps[u] > 0)
        llvm::errs() << "  " << uNames[u] << ": compute=" << unitLat[u]
                     << " blocked=" << unitStall[u] << " (" << unitOps[u]
                     << " ops)\n";
    }
    llvm::errs() << "\n";
  }
};
} // namespace

namespace mlir::buddy {
void registerReportCycleEstimatePass() {
  PassRegistration<ReportCycleEstimatePass>();
}
} // namespace mlir::buddy
