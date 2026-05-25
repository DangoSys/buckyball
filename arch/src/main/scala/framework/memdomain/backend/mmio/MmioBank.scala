package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.SramWriteIO
import framework.top.GlobalConfig

/**
 * MmioBank: single MMIO SRAM bank.
 *
 *   Internal storage : mmioBankEntries x mmioBankWidth bits (default 64 x 128 = 1 KB)
 *   Write port       : full mmioBankWidth (from DMA), reuses SramWriteIO for compatibility
 *   Read port        : mmioReadWidth bits (default 8), via byte_offset mux from 128-bit word
 *
 * Read latency : 1 cycle (SyncReadMem) + combinational byte-mux registered = 1 cycle
 * Write/read   : single-port semantics. Write takes priority when both valid.
 */
@instantiable
class MmioBank(val b: GlobalConfig) extends Module {

  private val numEntries  = b.memDomain.mmioBankEntries
  private val width       = b.memDomain.mmioBankWidth
  private val bytesPerRow = width / 8

  @public
  val io = IO(new Bundle {
    val write = new SramWriteIO(b)
    val read  = new MmioBankReadIO(b)
  })

  val mem = SyncReadMem(numEntries, UInt(width.W))

  io.write.req.ready := true.B
  io.read.req.ready  := !io.write.req.valid

  when(io.write.req.fire) {
    mem.write(io.write.req.bits.addr, io.write.req.bits.data)
  }
  io.write.resp.valid   := RegNext(io.write.req.fire, false.B)
  io.write.resp.bits.ok := RegNext(io.write.req.fire, false.B)

  val ren   = io.read.req.fire
  val rdata = mem.read(io.read.req.bits.addr, ren)

  val byteOffsetReg = RegNext(io.read.req.bits.byte_offset, 0.U)
  val byteSel       =
    (rdata >> (byteOffsetReg << 3.U))(b.memDomain.mmioReadWidth - 1, 0)

  io.read.resp.valid     := RegNext(ren, false.B)
  io.read.resp.bits.data := byteSel
}
