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

  val headerBuffers = (0 until b.top.ballMemChannelProducer).map(_ => Instantiate(new HeaderBuffer(b)))

  val freeChannels      = VecInit(headerBuffers.map(!_.io.read.lock))
  val freeChannelCount  = PopCount(freeChannels)
  val hasEnoughChannels = freeChannelCount >= io.peakChannelReq.bits.needed_channel_num

  val selectedChannelIds = Wire(Vec(b.top.ballMemChannelProducer, UInt(log2Up(b.top.ballMemChannelProducer).W)))
  val selectedMask       = Wire(Vec(b.top.ballMemChannelProducer, Bool()))

  val masks = Wire(Vec(b.top.ballMemChannelProducer + 1, Vec(b.top.ballMemChannelProducer, Bool())))
  masks(0) := freeChannels

  for (i <- 0 until b.top.ballMemChannelProducer) {
    val maskVec = masks(i)
    val selOH   = PriorityEncoderOH(maskVec.asUInt)
    val selIdx  = PriorityEncoder(maskVec.asUInt)
    selectedMask(i)       := maskVec.asUInt.orR && (i.U < io.peakChannelReq.bits.needed_channel_num)
    selectedChannelIds(i) := Mux(selectedMask(i), selIdx, 0.U)
    if (i < b.top.ballMemChannelProducer) {
      masks(i + 1) := VecInit(maskVec.zip(selOH.asBools).map { case (free, sel) => free && !sel })
    }
  }

  io.freeChannelResp.valid            := io.peakChannelReq.valid
  io.freeChannelResp.bits.is_free     := hasEnoughChannels
  io.freeChannelResp.bits.channel_ids := selectedChannelIds
  io.freeChannelResp.bits.channel_num := io.peakChannelReq.bits.needed_channel_num
  io.peakChannelReq.ready             := io.freeChannelResp.ready

  when(io.peakChannelReq.fire && hasEnoughChannels) {
    for (i <- 0 until b.top.ballMemChannelProducer) {
      val isSelected = selectedMask.zip(selectedChannelIds).map { case (sel, id) => sel && (id === i.U) }.reduce(_ || _)
      when(isSelected) {
        val headerData = Wire(new HeaderBufferData(b))
        headerData.lock               := true.B
        headerData.rob_id             := io.peakChannelReq.bits.rob_id
        headerData.bank_id            := io.peakChannelReq.bits.bank_id
        headerBuffers(i).io.set.valid := true.B
        headerBuffers(i).io.set.bits  := headerData
      }.otherwise {
        headerBuffers(i).io.set.valid := false.B
        headerBuffers(i).io.set.bits  := DontCare
      }
    }
  }.otherwise {
    for (i <- 0 until b.top.ballMemChannelProducer) {
      headerBuffers(i).io.set.valid := false.B
      headerBuffers(i).io.set.bits  := DontCare
    }
  }
}
