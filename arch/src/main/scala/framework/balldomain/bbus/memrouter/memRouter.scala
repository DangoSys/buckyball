package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.memrouter.Router

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  // balls - (readChannels/writeChannels) - bbus - (bbusChannel) - memdomain
  val bbusProducerChannels = b.ballDomain.bbusProducerChannels
  val bbusConsumerChannels = b.ballDomain.bbusConsumerChannels
  val numBalls             = b.ballDomain.ballNum
  val totalReadChannels    = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels   = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val maxPerChannelWidth   = b.ballDomain.ballIdMappings.flatMap(m => Seq(m.inBW, m.outBW)).max

  @public
  val io = IO(new Bundle {
    val bankRead_i  = Vec(totalReadChannels, new BankRead(b))
    val bankWrite_i = Vec(totalWriteChannels, new BankWrite(b))
    val bankRead_o  = Vec(bbusProducerChannels, Flipped(new BankRead(b)))
    val bankWrite_o = Vec(bbusProducerChannels, Flipped(new BankWrite(b)))
  })

  val readRouter:  Instance[Router] = Instantiate(new Router(totalReadChannels, bbusProducerChannels, maxPerChannelWidth))
  val writeRouter: Instance[Router] =
    Instantiate(new Router(totalWriteChannels, bbusProducerChannels, maxPerChannelWidth))

}
