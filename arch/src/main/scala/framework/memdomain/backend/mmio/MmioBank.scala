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

  val numEntries = b.memDomain.mmioBankEntries
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

  val readPending = RegNext(io.read.req.fire, false.B)
  val readHeld    = RegInit(false.B)
  val readData    = Reg(UInt(b.memDomain.mmioReadWidth.W))

  io.read.resp.valid := readPending || readHeld
  io.write.req.ready := true.B
  io.read.req.ready  := !io.write.req.valid && (!io.read.resp.valid || io.read.resp.ready)
  val ren = io.read.req.fire
  val wen = io.write.req.fire

  val rdata = mem.readWrite(
    Mux(wen, io.write.req.bits.addr, io.read.req.bits.addr),
    io.write.req.bits.data,
    ren || wen,
    wen
  )

  io.read.resp.bits.data := Mux(readHeld, readData, rdata)
  when(readPending && !io.read.resp.ready) {
    readHeld := true.B
    readData := rdata
  }.elsewhen(io.read.resp.fire) {
    readHeld := false.B
  }
}
