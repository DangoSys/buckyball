package memcore.bus.chi

import chisel3._

// Direction is from the transmitter. FLITV is a pulse, not ready/valid.
class Channel(val flitBits: Int) extends Bundle {
  require(flitBits >= 1)

  val flitpend = Output(Bool())
  val flitv    = Output(Bool())
  val flit     = Output(UInt(flitBits.W))
  val lcrdv    = Input(Bool())
}
