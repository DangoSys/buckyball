package framework.frontend.scoreboard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * BAT (Bank Alias Table)
 *
 * Rename virtual bank IDs into the high-ID alias namespace for scoreboard.
 * Lifetime policy: per ROB entry.
 *
 * ID partition:
 *   [0, vbankUpper]             : virtual/architected bank IDs
 *   [vbankUpper + 1, maxBankId] : renamed alias IDs (one alias per ROB entry)
 */
@instantiable
class BankAliasTable(val bankIdLen: Int, val vbankUpper: Int, val robEntries: Int) extends Module {
  private val aliasIdLen = bankIdLen
  private val robIdLen   = log2Up(robEntries)
  private val vbankNum   = vbankUpper + 1
  private val vbankIdLen = log2Up(vbankNum)
  private val maxBankId  = (1 << bankIdLen) - 1
  private val aliasBase  = vbankNum

  @public
  val io = IO(new Bundle {

    val alloc = Input(new Bundle {
      val valid  = Bool()
      val rob_id = UInt(robIdLen.W)
      val raw    = new BankAccessInfo(aliasIdLen)
    })

    val alloc_renamed = Output(new BankAccessInfo(aliasIdLen))

    val free = Input(new Bundle {
      val valid = Bool()
      val mask  = Vec(robEntries, Bool())
    })

  })

  // Current architectural alias for each virtual bank.
  val v2a = RegInit(VecInit((0 until vbankNum).map(_.U(aliasIdLen.W))))

  // Extra aliases [aliasBase, aliasBase + robEntries - 1] are one-per-ROB.
  val aliasInUse = RegInit(VecInit(Seq.fill(robEntries)(false.B)))

  // Per-entry metadata for commit-time free.
  val entHasWrite = RegInit(VecInit(Seq.fill(robEntries)(false.B)))
  val entOldAlias = RegInit(VecInit(Seq.fill(robEntries)(0.U(aliasIdLen.W))))
  val entNewAlias = RegInit(VecInit(Seq.fill(robEntries)(0.U(aliasIdLen.W))))
  val entWrVbank  = RegInit(VecInit(Seq.fill(robEntries)(0.U(vbankIdLen.W))))

  require(vbankUpper >= 0, s"BAT vbankUpper must be non-negative, got $vbankUpper")
  require(vbankUpper < maxBankId, s"BAT vbankUpper must be < maxBankId($maxBankId), got $vbankUpper")
  require(
    aliasBase + robEntries - 1 <= maxBankId,
    s"BAT alias range exceeds bank_id_len space: base=$aliasBase entries=$robEntries max=$maxBankId"
  )

  private def extraAlias(robId: UInt): UInt = aliasBase.U(aliasIdLen.W) + robId
  private def toVbankIdx(v:     UInt): UInt = v(vbankIdLen - 1, 0)

  private def mapVbank(v: UInt): UInt = {
    val idx = toVbankIdx(v)
    val out = WireDefault(0.U(aliasIdLen.W))
    for (i <- 0 until vbankNum) {
      when(idx === i.U(vbankIdLen.W)) {
        out := v2a(i)
      }
    }
    out
  }

  val q = io.alloc.raw
  io.alloc_renamed.rd_bank_0_valid := q.rd_bank_0_valid
  io.alloc_renamed.rd_bank_1_valid := q.rd_bank_1_valid
  io.alloc_renamed.wr_bank_valid   := q.wr_bank_valid
  io.alloc_renamed.rd_bank_0_id    := Mux(q.rd_bank_0_valid, mapVbank(q.rd_bank_0_id), 0.U)
  io.alloc_renamed.rd_bank_1_id    := Mux(q.rd_bank_1_valid, mapVbank(q.rd_bank_1_id), 0.U)
  io.alloc_renamed.wr_bank_id      := Mux(q.wr_bank_valid, extraAlias(io.alloc.rob_id), 0.U)

  when(io.alloc.valid) {
    val rid = io.alloc.rob_id
    when(q.rd_bank_0_valid) {
      assert(q.rd_bank_0_id <= vbankUpper.U, "BAT rd_bank_0_id exceeds virtual bank upper bound")
    }
    when(q.rd_bank_1_valid) {
      assert(q.rd_bank_1_id <= vbankUpper.U, "BAT rd_bank_1_id exceeds virtual bank upper bound")
    }
    when(q.wr_bank_valid) {
      assert(q.wr_bank_id <= vbankUpper.U, "BAT wr_bank_id exceeds virtual bank upper bound")
    }

    entHasWrite(rid) := q.wr_bank_valid
    entOldAlias(rid) := Mux(q.wr_bank_valid, mapVbank(q.wr_bank_id), 0.U)
    entNewAlias(rid) := Mux(q.wr_bank_valid, extraAlias(rid), 0.U)
    entWrVbank(rid)  := toVbankIdx(q.wr_bank_id)

    when(q.wr_bank_valid) {
      assert(!aliasInUse(rid), "BAT alias reused before free")
      aliasInUse(rid)               := true.B
      v2a(toVbankIdx(q.wr_bank_id)) := extraAlias(rid)
    }
  }

  when(io.free.valid) {
    for (i <- 0 until robEntries) {
      when(io.free.mask(i)) {
        when(entHasWrite(i)) {
          assert(aliasInUse(i), "BAT free on non-allocated alias")
          aliasInUse(i) := false.B

          // Restore old alias only if mapping still points to this entry's alias.
          when(v2a(entWrVbank(i)) === entNewAlias(i)) {
            v2a(entWrVbank(i)) := entOldAlias(i)
          }
        }
        entHasWrite(i) := false.B
        entOldAlias(i) := 0.U
        entNewAlias(i) := 0.U
        entWrVbank(i)  := 0.U
      }
    }
  }
}
