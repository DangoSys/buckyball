package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

class ChannelClusterIO(val b: GlobalConfig, numChannels: Int) extends Bundle {
  val channelIn       = Vec(numChannels, Flipped(new ChannelIO(b)))
  val channelOut      = Vec(numChannels, new ChannelIO(b))
  val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
  val freeChannelResp = Decoupled(new FreeChannelResp(b))
}

@instantiable
class ChannelCluster(val b: GlobalConfig, numChannels: Int) extends Module {
  @public
  val io = IO(new ChannelClusterIO(b, numChannels))

  val channels = (0 until numChannels).map(_ => Instantiate(new Channel(b)))

  // Connect channels
  for (i <- 0 until numChannels) {
    channels(i).io.in <> io.channelIn(i)
    channels(i).io.out <> io.channelOut(i)
    channels(i).io.peakChannelReq.valid  := io.peakChannelReq.valid
    channels(i).io.peakChannelReq.bits   := io.peakChannelReq.bits
    channels(i).io.freeChannelResp.ready := true.B
  }

  // Aggregate free channel responses
  val freeChannels      = VecInit(channels.map(ch => ch.io.freeChannelResp.bits.is_free))
  val freeChannelCount  = PopCount(freeChannels)
  val hasEnoughChannels = freeChannelCount >= io.peakChannelReq.bits.needed_channel_num

  val selectedChannelIds = Wire(Vec(numChannels, UInt(log2Up(numChannels).W)))
  val selectedMask       = Wire(Vec(numChannels, Bool()))

  val masks = Wire(Vec(numChannels + 1, Vec(numChannels, Bool())))
  masks(0) := freeChannels

  for (i <- 0 until numChannels) {
    val maskVec = masks(i)
    val selOH   = PriorityEncoderOH(maskVec.asUInt)
    val selIdx  = PriorityEncoder(maskVec.asUInt)
    selectedMask(i)       := maskVec.asUInt.orR && (i.U < io.peakChannelReq.bits.needed_channel_num)
    selectedChannelIds(i) := Mux(selectedMask(i), selIdx, 0.U)
    if (i < numChannels) {
      masks(i + 1) := VecInit(maskVec.zip(selOH.asBools).map { case (free, sel) => free && !sel })
    }
  }

  io.freeChannelResp.valid            := io.peakChannelReq.valid
  io.freeChannelResp.bits.is_free     := hasEnoughChannels
  io.freeChannelResp.bits.channel_ids := selectedChannelIds
  io.freeChannelResp.bits.channel_num := io.peakChannelReq.bits.needed_channel_num
  io.peakChannelReq.ready             := io.freeChannelResp.ready
}
