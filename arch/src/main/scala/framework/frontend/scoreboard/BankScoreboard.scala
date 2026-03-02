package framework.frontend.scoreboard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * Bank access information extracted from instruction encoding.
 * This is instruction-agnostic — the scoreboard only sees read/write bank sets.
 */
class BankAccessInfo(val bankIdLen: Int) extends Bundle {
  val rd_bank_0_valid = Bool()
  val rd_bank_0_id    = UInt(bankIdLen.W)
  val rd_bank_1_valid = Bool()
  val rd_bank_1_id    = UInt(bankIdLen.W)
  val wr_bank_valid   = Bool()
  val wr_bank_id      = UInt(bankIdLen.W)
}

object BankAccessInfo {

  /** Create a zero-valued (no access) BankAccessInfo */
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

/**
 * Bank Scoreboard — tracks in-flight read/write operations per bank.
 *
 * Hazard rules:
 *   - Read bank X  → requires bankWrBusy(X) == false       (RAW)
 *   - Write bank X → requires bankRdCount(X) == 0           (WAR)
 *                     AND     bankWrBusy(X) == false         (WAW)
 *
 * bankRdCount: multi-bit counter (multiple concurrent readers allowed, RR is OK)
 * bankWrBusy:  1-bit flag (WAW rule guarantees at most 1 writer in-flight)
 */
@instantiable
class BankScoreboard(val bankNum: Int, val robEntries: Int) extends Module {

  val bankIdLen = log2Up(bankNum)
  val cntWidth  = log2Ceil(robEntries + 1)

  // Issue: increment counters for the accessed banks
  @public
  val issue     = IO(Flipped(Valid(new BankAccessInfo(bankIdLen))))
  // Complete: decrement counters for the accessed banks
  @public
  val complete  = IO(Flipped(Valid(new BankAccessInfo(bankIdLen))))
  // Hazard query: check if the given access would conflict
  @public
  val query     = IO(Input(new BankAccessInfo(bankIdLen)))
  @public
  val hasHazard = IO(Output(Bool()))

  // Read counter: multi-bit, allows concurrent readers
  val bankRdCount = RegInit(VecInit(Seq.fill(bankNum)(0.U(cntWidth.W))))
  // Write flag: 1-bit, WAW ensures at most 1 writer in-flight per bank
  val bankWrBusy  = RegInit(VecInit(Seq.fill(bankNum)(false.B)))

  // --- Hazard detection ---
  val q          = query
  val rd0_hazard = q.rd_bank_0_valid && bankWrBusy(q.rd_bank_0_id)
  val rd1_hazard = q.rd_bank_1_valid && bankWrBusy(q.rd_bank_1_id)

  val wr_hazard = q.wr_bank_valid && (
    bankRdCount(q.wr_bank_id) =/= 0.U ||
      bankWrBusy(q.wr_bank_id)
  )

  hasHazard := rd0_hazard || rd1_hazard || wr_hazard

  // --- Issue: increment counters ---
  when(issue.valid) {
    val info = issue.bits
    when(info.rd_bank_0_valid) {
      bankRdCount(info.rd_bank_0_id) := bankRdCount(info.rd_bank_0_id) + 1.U
    }
    when(info.rd_bank_1_valid) {
      bankRdCount(info.rd_bank_1_id) := bankRdCount(info.rd_bank_1_id) + 1.U
    }
    when(info.wr_bank_valid) {
      bankWrBusy(info.wr_bank_id) := true.B
    }
  }

  // --- Complete: decrement counters ---
  when(complete.valid) {
    val info = complete.bits
    when(info.rd_bank_0_valid) {
      bankRdCount(info.rd_bank_0_id) := bankRdCount(info.rd_bank_0_id) - 1.U
    }
    when(info.rd_bank_1_valid) {
      bankRdCount(info.rd_bank_1_id) := bankRdCount(info.rd_bank_1_id) - 1.U
    }
    when(info.wr_bank_valid) {
      bankWrBusy(info.wr_bank_id) := false.B
    }
  }
}
