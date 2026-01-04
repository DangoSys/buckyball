package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * SramBank: Pure SRAM bank
 * Simple read/write memory without any accumulation logic
 * Each bank is a single-port SRAM
 */
@instantiable
class SramBank(val b: GlobalConfig) extends Module {
  val mask_len  = b.memDomain.bankMaskLen
  val mask_elem = UInt((b.memDomain.bankWidth / mask_len).W)

  @public
  val io = IO(new Bundle {
    val sramRead  = new SramReadIO(b)
    val sramWrite = new SramWriteIO(b)
  })

  val mem = SyncReadMem(b.memDomain.bankEntries, Vec(mask_len, mask_elem))

  // -----------------------------------------------------------------------------
  // Read path
  // -----------------------------------------------------------------------------
  io.sramRead.req.ready := !io.sramWrite.req.valid

  val raddr = io.sramRead.req.bits.addr
  val ren   = io.sramRead.req.fire
  val rdata = mem.read(raddr, ren)

  io.sramRead.resp.valid     := RegNext(ren)
  io.sramRead.resp.bits.data := RegNext(rdata.asUInt)

  // -----------------------------------------------------------------------------
  // Write path
  // -----------------------------------------------------------------------------
  io.sramWrite.req.ready := !io.sramRead.req.valid

  when(io.sramWrite.req.valid) {
    mem.write(
      io.sramWrite.req.bits.addr,
      io.sramWrite.req.bits.data.asTypeOf(Vec(mask_len, mask_elem)),
      io.sramWrite.req.bits.mask
    )
  }

  io.sramWrite.resp.valid   := RegNext(io.sramWrite.req.fire)
  io.sramWrite.resp.bits.ok := RegNext(io.sramWrite.req.fire)
}
