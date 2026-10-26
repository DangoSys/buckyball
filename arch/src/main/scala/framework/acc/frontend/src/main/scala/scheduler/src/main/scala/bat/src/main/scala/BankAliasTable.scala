package framework.acc.frontend.scheduler.bat

import framework.acc.frontend.scheduler.scoreboard.BankAccessInfo
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.frontend.scheduler.bat.configs.Params

/** Current mappings and a free list; overwritten aliases return only at retirement. */
@instantiable
class BankAliasTable(val p: Params) extends Module {

  val bankIdLen  = p.bankIdBits
  val vbankNum   = p.virtualBanks
  val robEntries = p.robEntries
  val aliasCount = 2 * vbankNum + robEntries

  @public
  val io = IO(new Bundle {

    val alloc = Input(new Bundle {
      val valid       = Bool()
      val rename      = Bool()
      val reset_read0 = Bool()
      val raw         = new BankAccessInfo(p.bankIdBits)
    })

    val ready         = Output(Bool())
    val alloc_renamed = Output(new BankAccessInfo(p.bankIdBits))
    val reclaim       = Flipped(Valid(UInt(bankIdLen.W)))
    val stale         = Output(Vec(2, Valid(UInt(bankIdLen.W))))
  })

  val current = RegInit(VecInit((0 until vbankNum).map(_.U(bankIdLen.W))))

  val initialFree = ((BigInt(1) << aliasCount) - 1) ^ ((BigInt(1) << vbankNum) - 1)
  val free        = RegInit(initialFree.U(aliasCount.W))
  val available   = free

  val fresh = PriorityEncoder(available)
  val raw   = io.alloc.raw

  // Architectural IDs are checked before allocation; the register array uses only its index bits.
  def currentAlias(bank: UInt): UInt =
    if (vbankNum == 1) current(0) else current(bank(log2Ceil(vbankNum) - 1, 0))

  io.ready                      := !io.alloc.rename || available.orR
  io.alloc_renamed              := raw
  io.alloc_renamed.rd_bank_0_id := Mux(
    raw.rd_bank_0_valid,
    currentAlias(raw.rd_bank_0_id),
    0.U
  )
  io.alloc_renamed.rd_bank_1_id := Mux(
    raw.rd_bank_1_valid,
    currentAlias(raw.rd_bank_1_id),
    0.U
  )
  io.alloc_renamed.rd_wr_id     := Mux(raw.rd_wr_valid, currentAlias(raw.rd_wr_id), 0.U)
  io.alloc_renamed.wr_bank_id   := Mux(
    io.alloc.rename,
    fresh,
    Mux(raw.wr_update, currentAlias(raw.wr_bank_id), raw.wr_bank_id)
  )
  val resetBank   = raw.rd_bank_0_id
  val resetsWrite = raw.wr_bank_valid && !io.alloc.rename && !raw.wr_update &&
    currentAlias(raw.wr_bank_id) =/= raw.wr_bank_id
  io.stale(0).valid := io.alloc.rename || (!io.alloc.reset_read0 && resetsWrite) ||
    (io.alloc.reset_read0 && currentAlias(resetBank) =/= resetBank)
  io.stale(0).bits  := Mux(io.alloc.reset_read0, currentAlias(resetBank), currentAlias(raw.wr_bank_id))
  io.stale(1).valid := io.alloc.reset_read0 && resetsWrite && resetBank =/= raw.wr_bank_id
  io.stale(1).bits  := currentAlias(raw.wr_bank_id)

  when(io.reclaim.valid)(assert(io.reclaim.bits < aliasCount.U, "BAT reclaim alias is outside the allocated namespace"))

  val returned  =
    Mux(io.reclaim.valid && io.reclaim.bits >= vbankNum.U, UIntToOH(io.reclaim.bits, aliasCount), 0.U(aliasCount.W))
  val allocated = Mux(io.alloc.valid && io.alloc.rename, UIntToOH(fresh, aliasCount), 0.U(aliasCount.W))
  free := (free | returned) & ~allocated

  when(io.alloc.valid) {
    assert(io.ready, "BAT allocation has no free virtual bank")
    when(raw.rd_bank_0_valid || io.alloc.reset_read0)(assert(
      raw.rd_bank_0_id < vbankNum.U,
      "BAT read bank is outside the architectural namespace"
    ))
    when(raw.rd_bank_1_valid)(assert(
      raw.rd_bank_1_id < vbankNum.U,
      "BAT second read bank is outside the architectural namespace"
    ))
    when(raw.rd_wr_valid)(assert(raw.rd_wr_id < vbankNum.U, "BAT update source is outside the architectural namespace"))
    when(raw.wr_bank_valid || io.alloc.rename)(assert(
      raw.wr_bank_id < vbankNum.U,
      "BAT write bank is outside the architectural namespace"
    ))
    val access = io.alloc_renamed
    when(raw.wr_bank_valid) {
      if (vbankNum == 1) current(0)                           := access.wr_bank_id
      else current(raw.wr_bank_id(log2Ceil(vbankNum) - 1, 0)) := access.wr_bank_id
    }
    when(io.alloc.reset_read0) {
      if (vbankNum == 1) current(0)                             := raw.rd_bank_0_id
      else current(raw.rd_bank_0_id(log2Ceil(vbankNum) - 1, 0)) := raw.rd_bank_0_id
    }
  }
}
