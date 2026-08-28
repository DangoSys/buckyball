package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/**
 * MmioBank: single MMIO SRAM bank.
 *
 *   Internal storage : mmioBankEntries x mmioReadWidth bits (default 1024 x 8 = 1 KB)
 *   Write port       : one byte; MmioPool distributes DMA beats across banks.
 *   Read port        : one byte.
 *
 * Read latency : 1 cycle. Write/read uses single-port semantics; write wins.
 */
@instantiable
class MmioBank(val b: GlobalConfig) extends Module {

  private val numEntries = b.memDomain.mmioBankEntries
  require(
    b.memDomain.mmioBankWidth == b.memDomain.mmioReadWidth,
    "MmioBank requires one physical MMIO element per read"
  )
  require(
    b.memDomain.mmioReadWidth == 8,
    "MmioBank requires 8-bit MMIO elements"
  )

  @public
  val io = IO(new Bundle {
    val write = new MmioBankWriteIO(b)
    val read  = new MmioBankReadIO(b)
  })

  val mem = SyncReadMem(numEntries, UInt(b.memDomain.mmioReadWidth.W))

  io.write.req.ready := true.B
  io.read.req.ready  := !io.write.req.valid

  when(io.write.req.fire) {
    mem.write(io.write.req.bits.addr, io.write.req.bits.data)
  }

  val ren   = io.read.req.fire
  val rdata = mem.read(io.read.req.bits.addr, ren)

  io.read.resp.valid     := RegNext(ren, false.B)
  io.read.resp.bits.data := rdata
}
