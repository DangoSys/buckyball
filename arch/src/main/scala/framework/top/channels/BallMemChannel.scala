package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

class BallMemChannelIO(val b: GlobalConfig) extends Bundle {
  val channelToMemDomain   = Vec(b.top.ballMemChannelProducer, new ChannelIO(b))
  val channelFromMemDomain = Vec(b.top.ballMemChannelConsumer, Flipped(new ChannelIO(b)))

  val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
  val freeChannelResp = Decoupled(new FreeChannelResp(b))
}

@instantiable
class BallMemChannel(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new BallMemChannelIO(b))

  val channelsToMem   = (0 until b.top.ballMemChannelProducer).map(_ => Instantiate(new Channel(b)))
  val channelsFromMem = (0 until b.top.ballMemChannelConsumer).map(_ => Instantiate(new Channel(b)))

  VecInit(channelsToMem.map(_.io)) <> io.channelToMemDomain
  VecInit(channelsFromMem.map(_.io)) <> io.channelFromMemDomain

  val freeChannelCount = PopCount(io.channelToMemDomain.map(!_.header.read.lock))
  io.freeChannelResp.valid        := io.peakChannelReq.valid
  io.freeChannelResp.bits.is_free := freeChannelCount >= io.peakChannelReq.bits.needed_channel_num
  io.peakChannelReq.ready         := io.freeChannelResp.ready
}
