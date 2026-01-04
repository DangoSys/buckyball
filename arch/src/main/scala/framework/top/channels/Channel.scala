package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.top.channels.{HeaderBuffer, HeaderBufferData}

class ChannelIO(val b: GlobalConfig) extends Bundle {
  val producer = Flipped(Decoupled(UInt(b.memDomain.bankWidth.W)))
  val consumer = Decoupled(UInt(b.memDomain.bankWidth.W))

  val header = new Bundle {
    val read = Output(new HeaderBufferData(b))
    val set  = Flipped(Decoupled(new HeaderBufferData(b)))
  }

}

@instantiable
class Channel(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new ChannelIO(b))

  val headerBuffer: Instance[HeaderBuffer] = Instantiate(new HeaderBuffer(b))

  io.header.read := headerBuffer.io.read
  headerBuffer.io.set <> io.header.set

  io.consumer <> io.producer
}
