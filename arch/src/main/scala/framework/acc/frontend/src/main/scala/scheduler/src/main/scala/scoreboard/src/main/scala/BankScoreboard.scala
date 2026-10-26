package framework.acc.frontend.scheduler.scoreboard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.frontend.scheduler.scoreboard.configs.Params

//====----------------------------------------------------------====//
//  Bank Scoreboard: tracks in-flight read/write operations per bank.
//
//  Hazard rules:
//   - Read bank X  → requires bankWrBusy(X) == false       (RAW)
//   - Write bank X → requires bankRdCount(X) == 0           (WAR)
//                     AND     bankWrBusy(X) == false         (WAW)
//
//  bankRdCount: multi-bit counter (multiple concurrent readers allowed, RR is OK)
//  bankWrBusy:  1-bit flag (WAW rule guarantees at most 1 writer in-flight)
//
//  Lifecycle:
//   - alloc:    sets bankWrBusy for the renamed write alias. This must happen
//               at alloc time (not issue) because the BAT updates v2a at alloc,
//               and subsequent reads of the same vbank will be renamed to the
//               new alias immediately. If we only set bankWrBusy at issue, the
//               window between alloc and issue lets dependent reads slip through
//               the hazard check and issue out-of-order.
//   - issue:    increments bankRdCount for the renamed read aliases.
//   - complete: decrements bankRdCount and clears bankWrBusy.
//
//  Issue and complete may fire in the same cycle. Updates are computed
//  combinationally so both increments and decrements take effect.
//====----------------------------------------------------------====//

@instantiable
class BankScoreboard(val p: Params) extends Module {

  val bankIdLen  = p.bankIdBits
  val bankNum    = p.banks
  val robEntries = p.entries

  val cntWidth = log2Ceil(3 * robEntries + 1)

  @public
  val io = IO(new Bundle {
    val alloc    = Flipped(Valid(new BankAccessInfo(p.bankIdBits)))
    val issue    = Flipped(Valid(new BankAccessInfo(p.bankIdBits)))
    val complete = Flipped(Valid(new BankAccessInfo(p.bankIdBits)))

    val queryVec = Input(Vec(
      robEntries,
      new BankAccessInfo(p.bankIdBits)
    ))

    val hazardVec = Output(Vec(robEntries, Bool()))
  })

  val bankRdCount = RegInit(VecInit(Seq.fill(bankNum)(0.U(cntWidth.W))))
  val bankWrBusy  = RegInit(VecInit(Seq.fill(bankNum)(false.B)))
  def bankIndex(id: UInt): UInt = id(log2Ceil(bankNum) - 1, 0)

  // --- Hazard detection (reads current register state) ---
  def hazardOf(q: BankAccessInfo): Bool = {
    val rd0_hazard = q.rd_bank_0_valid && bankWrBusy(bankIndex(q.rd_bank_0_id))
    val rd1_hazard = q.rd_bank_1_valid && bankWrBusy(bankIndex(q.rd_bank_1_id))
    // Write hazard is NOT checked here. BAT eliminates WAW/WAR by giving each
    // write a unique alias that no other instruction uses. Checking wr_hazard
    // would falsely block the writer itself, since io.alloc marks the writer's
    // own alias as busy in the same cycle the entry enters the ROB.
    val rdWrHazard = q.rd_wr_valid && bankWrBusy(bankIndex(q.rd_wr_id))
    rd0_hazard || rd1_hazard || rdWrHazard
  }

  for (i <- 0 until robEntries) {
    io.hazardVec(i) := hazardOf(io.queryVec(i))
  }

  // --- Compute per-bank deltas to handle simultaneous io.issue + io.complete ---
  for (bank <- 0 until bankNum) {
    val bankU = bank.U(bankIdLen.W)

    // Read counter: +1 per io.issue read, -1 per io.complete read
    val issRd0  =
      io.issue.valid && io.issue.bits.rd_bank_0_valid && (io.issue.bits.rd_bank_0_id === bankU)
    val issRd1  =
      io.issue.valid && io.issue.bits.rd_bank_1_valid && (io.issue.bits.rd_bank_1_id === bankU)
    val cmpRd0  =
      io.complete.valid && io.complete.bits.rd_bank_0_valid && (io.complete.bits.rd_bank_0_id === bankU)
    val cmpRd1  =
      io.complete.valid && io.complete.bits.rd_bank_1_valid && (io.complete.bits.rd_bank_1_id === bankU)
    val issRdWr =
      io.issue.valid && io.issue.bits.rd_wr_valid && io.issue.bits.rd_wr_id === bankU
    val cmpRdWr =
      io.complete.valid && io.complete.bits.rd_wr_valid && io.complete.bits.rd_wr_id === bankU

    val rdInc = issRd0.asUInt +& issRd1.asUInt +& issRdWr.asUInt
    val rdDec = cmpRd0.asUInt +& cmpRd1.asUInt +& cmpRdWr.asUInt
    bankRdCount(bank) := bankRdCount(bank) + rdInc - rdDec

    // Write flag: io.alloc sets (alias is reserved as soon as it is renamed),
    // io.complete clears. If both happen to same bank, io.complete takes priority.
    val allocWr =
      io.alloc.valid && io.alloc.bits.wr_bank_valid && (io.alloc.bits.wr_bank_id === bankU)
    val issueWr =
      io.issue.valid && io.issue.bits.wr_bank_valid && (io.issue.bits.wr_bank_id === bankU)
    val cmpWr   =
      io.complete.valid && io.complete.bits.wr_bank_valid && (io.complete.bits.wr_bank_id === bankU)

    when((allocWr || issueWr) && !cmpWr) {
      bankWrBusy(bank) := true.B
    }.elsewhen(cmpWr) {
      bankWrBusy(bank) := false.B
    }
  }
}
