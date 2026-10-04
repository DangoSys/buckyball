package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.util._

class DmaReadCommand extends Bundle {
  val transfer = new BBReadRequest
  // Indices of MemFrontend's physical footprints: loader=0, storer=1, kernel image=2.
  val producer = UInt(2.W)
}

/** Accelerator-side logical transfers. Translation and external-memory transport belong to the system. */
class DmaPort(dataBits: Int) extends Bundle {
  val read        = Decoupled(new DmaReadCommand)
  val readResult  = Flipped(Decoupled(new BBReadResponse(dataBits)))
  val write       = Decoupled(new BBWriteCommand)
  val writeData   = Decoupled(new BBWriteData(dataBits))
  val writeResult = Flipped(Decoupled(new BBWriteResponse))
  val readBusy    = Input(Bool())
  val writeBusy   = Input(Bool())
}
