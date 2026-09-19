package memcore.bus.axi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class MasterIO(
  dataBits: Int,
  idBits:   Int,
  destBits: Int,
  userBits: Int)
    extends Bundle {
  val in   = Flipped(Decoupled(new Beat(dataBits, idBits, destBits, userBits)))
  val axis = new Port(dataBits, idBits, destBits, userBits)
}

/** Drives a physical AXI4-Stream master port from an internal stream. */
@instantiable
class Master(
  dataBits: Int,
  idBits:   Int = 0,
  destBits: Int = 0,
  userBits: Int = 0)
    extends Module {

  @public
  val io = IO(new MasterIO(dataBits, idBits, destBits, userBits))

  io.axis.tvalid := io.in.valid
  io.in.ready    := io.axis.tready
  io.axis.tdata  := io.in.bits.tdata
  io.axis.tkeep  := io.in.bits.tkeep
  io.axis.tlast  := io.in.bits.tlast
  io.axis.tid    := io.in.bits.tid
  io.axis.tdest  := io.in.bits.tdest
  io.axis.tuser  := io.in.bits.tuser
}
