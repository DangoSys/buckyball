package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

class ChannelIO(val b: GlobalConfig) extends Bundle {
  val producer = Flipped(Decoupled(UInt(b.memDomain.bankWidth.W)))
  val consumer = Decoupled(UInt(b.memDomain.bankWidth.W))
}

@instantiable
class Channel(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new ChannelIO(b))

  io.consumer <> io.producer
}
