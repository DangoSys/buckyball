package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * SramBank: Pure SRAM bank
 * Simple read/write memory without any accumulation logic
 * Each bank is a single-port SRAM. `clear` zeroes every row locally, one row per cycle;
 * the ports refuse requests while `clearing`, so banks clear in parallel.
 */
@instantiable
class SramBank(val b: GlobalConfig) extends Module {
  val mask_len  = b.memDomain.bankMaskLen
  val mask_elem = UInt((b.memDomain.bankWidth / mask_len).W)

  @public
  val io = IO(new Bundle {
    val sramRead  = new SramReadIO(b)
    val sramWrite = new SramWriteIO(b)
    val clear     = Input(Bool())
    val clearing  = Output(Bool())
  })

  val mem = SyncReadMem(b.memDomain.bankEntries, Vec(mask_len, mask_elem))

  // -----------------------------------------------------------------------------
  // Local clear
  // -----------------------------------------------------------------------------
  val clearing = RegInit(false.B)
  val clearRow = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  io.clearing := clearing
  when(io.clear) {
    clearing := true.B
    clearRow := 0.U
  }.elsewhen(clearing) {
    clearRow                                                    := clearRow + 1.U
    when(clearRow === (b.memDomain.bankEntries - 1).U)(clearing := false.B)
  }

  // -----------------------------------------------------------------------------
  // Read path
  // -----------------------------------------------------------------------------
  io.sramRead.req.ready := !io.sramWrite.req.valid && !clearing

  val raddr = io.sramRead.req.bits.addr
  val ren   = io.sramRead.req.fire
  val rdata = mem.read(raddr, ren)

  io.sramRead.resp.valid     := RegNext(ren)
  io.sramRead.resp.bits.data := rdata.asUInt

  // -----------------------------------------------------------------------------
  // Write path
  // -----------------------------------------------------------------------------
  io.sramWrite.req.ready := !io.sramRead.req.valid && !clearing

  // One write port serves both requests and the local clear.
  when(io.sramWrite.req.fire || clearing) {
    mem.write(
      Mux(clearing, clearRow, io.sramWrite.req.bits.addr),
      Mux(clearing, 0.U.asTypeOf(Vec(mask_len, mask_elem)), io.sramWrite.req.bits.data.asTypeOf(Vec(mask_len, mask_elem))),
      Mux(clearing, VecInit(Seq.fill(mask_len)(true.B)), io.sramWrite.req.bits.mask)
    )
  }

  io.sramWrite.resp.valid   := RegNext(io.sramWrite.req.fire)
  io.sramWrite.resp.bits.ok := RegNext(io.sramWrite.req.fire)
}
