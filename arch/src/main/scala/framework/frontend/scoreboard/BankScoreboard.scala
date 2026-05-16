package framework.frontend.scoreboard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

//====----------------------------------------------------------====//
//  Bank access information extracted from instruction encoding.
//  It's instruction-agnostic: the scoreboard only sees read/write bank_id.
//====----------------------------------------------------------====//
class BankAccessInfo(val bankIdLen: Int) extends Bundle {
  val rd_bank_0_valid = Bool()
  val rd_bank_0_id    = UInt(bankIdLen.W)
  val rd_bank_1_valid = Bool()
  val rd_bank_1_id    = UInt(bankIdLen.W)
  val wr_bank_valid   = Bool()
  val wr_bank_id      = UInt(bankIdLen.W)
}

object BankAccessInfo {

  def none(bankIdLen: Int): BankAccessInfo = {
    val w = Wire(new BankAccessInfo(bankIdLen))
    w.rd_bank_0_valid := false.B
    w.rd_bank_0_id    := 0.U
    w.rd_bank_1_valid := false.B
    w.rd_bank_1_id    := 0.U
    w.wr_bank_valid   := false.B
    w.wr_bank_id      := 0.U
    w
  }

}

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
class BankScoreboard(val bankNum: Int, val robEntries: Int) extends Module {

  val bankIdLen = log2Up(bankNum)
  val cntWidth  = log2Ceil(robEntries + 1)

  @public
  val alloc     = IO(Flipped(Valid(new BankAccessInfo(bankIdLen))))
  @public
  val issue     = IO(Flipped(Valid(new BankAccessInfo(bankIdLen))))
  @public
  val complete  = IO(Flipped(Valid(new BankAccessInfo(bankIdLen))))
  @public
  val query     = IO(Input(new BankAccessInfo(bankIdLen)))
  @public
  val hasHazard = IO(Output(Bool()))
  @public
  val queryVec  = IO(Input(Vec(robEntries, new BankAccessInfo(bankIdLen))))
  @public
  val hazardVec = IO(Output(Vec(robEntries, Bool())))

  val bankRdCount = RegInit(VecInit(Seq.fill(bankNum)(0.U(cntWidth.W))))
  val bankWrBusy  = RegInit(VecInit(Seq.fill(bankNum)(false.B)))

  // --- Hazard detection (reads current register state) ---
  private def hazardOf(q: BankAccessInfo): Bool = {
    val rd0_hazard = q.rd_bank_0_valid && bankWrBusy(q.rd_bank_0_id)
    val rd1_hazard = q.rd_bank_1_valid && bankWrBusy(q.rd_bank_1_id)
    // Write hazard is NOT checked here. BAT eliminates WAW/WAR by giving each
    // write a unique alias that no other instruction uses. Checking wr_hazard
    // would falsely block the writer itself, since alloc marks the writer's
    // own alias as busy in the same cycle the entry enters the ROB.
    rd0_hazard || rd1_hazard
  }

  hasHazard := hazardOf(query)
  for (i <- 0 until robEntries) {
    hazardVec(i) := hazardOf(queryVec(i))
  }

  // --- Compute per-bank deltas to handle simultaneous issue + complete ---
  for (bank <- 0 until bankNum) {
    val bankU = bank.U(bankIdLen.W)

    // Read counter: +1 per issue read, -1 per complete read
    val issRd0 = issue.valid && issue.bits.rd_bank_0_valid && (issue.bits.rd_bank_0_id === bankU)
    val issRd1 = issue.valid && issue.bits.rd_bank_1_valid && (issue.bits.rd_bank_1_id === bankU)
    val cmpRd0 = complete.valid && complete.bits.rd_bank_0_valid && (complete.bits.rd_bank_0_id === bankU)
    val cmpRd1 = complete.valid && complete.bits.rd_bank_1_valid && (complete.bits.rd_bank_1_id === bankU)

    val rdInc = issRd0.asUInt +& issRd1.asUInt
    val rdDec = cmpRd0.asUInt +& cmpRd1.asUInt
    bankRdCount(bank) := bankRdCount(bank) + rdInc - rdDec

    // Write flag: alloc sets (alias is reserved as soon as it is renamed),
    // complete clears. If both happen to same bank, complete takes priority.
    val allocWr = alloc.valid && alloc.bits.wr_bank_valid && (alloc.bits.wr_bank_id === bankU)
    val cmpWr   = complete.valid && complete.bits.wr_bank_valid && (complete.bits.wr_bank_id === bankU)

    when(allocWr && !cmpWr) {
      bankWrBusy(bank) := true.B
    }.elsewhen(cmpWr) {
      bankWrBusy(bank) := false.B
    }
  }
}
