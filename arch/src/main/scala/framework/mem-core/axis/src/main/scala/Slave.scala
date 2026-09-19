package memcore.bus.axi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class SlaveIO(
  dataBits: Int,
  idBits:   Int,
  destBits: Int,
  userBits: Int)
    extends Bundle {
  val axis = Flipped(new Port(dataBits, idBits, destBits, userBits))
  val out  = Decoupled(new Beat(dataBits, idBits, destBits, userBits))
}

/** Receives a physical AXI4-Stream slave port as an internal stream. */
@instantiable
class Slave(
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Module {

  @public
  val io = IO(new SlaveIO(dataBits, idBits, destBits, userBits))

  io.out.valid      := io.axis.tvalid
  io.axis.tready    := io.out.ready
  io.out.bits.tdata := io.axis.tdata
  io.out.bits.tkeep := io.axis.tkeep
  io.out.bits.tlast := io.axis.tlast
  io.out.bits.tid   := io.axis.tid
  io.out.bits.tdest := io.axis.tdest
  io.out.bits.tuser := io.axis.tuser
}
