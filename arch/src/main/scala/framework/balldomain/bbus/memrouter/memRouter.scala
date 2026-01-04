package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

class PeakChannelReq(val b: GlobalConfig) extends Bundle {
  val needed_channel_num = UInt(log2Up(b.ballDomain.ballIdMappings.map(_.inBW).sum + 1).W)
}

class FreeChannelResp(val b: GlobalConfig) extends Bundle {
  val is_free = Bool()
}

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val numBalls             = b.ballDomain.ballNum
  val bbusProducerChannels = b.ballDomain.bbusProducerChannels
  val bbusConsumerChannels = b.ballDomain.bbusConsumerChannels
  val totalReadChannels    = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels   = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val maxPerChannelWidth   = b.ballDomain.ballIdMappings.flatMap(m => Seq(m.inBW, m.outBW)).max

  @public
  val io = IO(new Bundle {
    val bankRead_i = Vec(totalReadChannels, new BankRead(b))
    val bankRead_o = Vec(bbusProducerChannels, Flipped(new BankRead(b)))

    val bankWrite_i = Vec(totalWriteChannels, new BankWrite(b))
    val bankWrite_o = Vec(bbusProducerChannels, Flipped(new BankWrite(b)))

    val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
    val freeChannelResp = Decoupled(new FreeChannelResp(b))
  })

  val readReqGen: Instance[ReadReqGen] = Instantiate(new ReadReqGen(b))
  readReqGen.io.bank_read_i <> io.bankRead_i

  val hasReadReq    = readReqGen.io.read_req_o.valid
  val isChannelFree = io.freeChannelResp.valid && io.freeChannelResp.bits.is_free
  val reqAccepted   = isChannelFree

  io.peakChannelReq.valid                   := hasReadReq
  io.peakChannelReq.bits.needed_channel_num := readReqGen.io.read_req_o.bits.channel_num
  io.freeChannelResp.ready                  := hasReadReq

  readReqGen.io.read_req_o.ready := reqAccepted
}
