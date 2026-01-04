package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

class ChannelIO(val b: GlobalConfig) extends Bundle {
  val data = Decoupled(UInt(b.memDomain.bankWidth.W))
}

@instantiable
class Channel(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val in  = Flipped(new ChannelIO(b))
    val out = new ChannelIO(b)

    val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
    val freeChannelResp = Decoupled(new FreeChannelResp(b))
  })

  val headerBuffer: Instance[HeaderBuffer] = Instantiate(new HeaderBuffer(b))

  val isFree           = !headerBuffer.io.read.lock
  val hasEnoughChannel = isFree

  io.freeChannelResp.valid            := io.peakChannelReq.valid
  io.freeChannelResp.bits.is_free     := hasEnoughChannel
  io.freeChannelResp.bits.channel_ids := DontCare
  io.freeChannelResp.bits.channel_num := io.peakChannelReq.bits.needed_channel_num
  io.peakChannelReq.ready             := io.freeChannelResp.ready

  when(io.peakChannelReq.fire && hasEnoughChannel) {
    headerBuffer.io.set.valid        := true.B
    headerBuffer.io.set.bits.rob_id  := io.peakChannelReq.bits.rob_id
    headerBuffer.io.set.bits.bank_id := io.peakChannelReq.bits.bank_id
    headerBuffer.io.set.bits.lock    := true.B
  }.otherwise {
    headerBuffer.io.set.valid := false.B
    headerBuffer.io.set.bits  := DontCare
  }

  io.out <> io.in
}
