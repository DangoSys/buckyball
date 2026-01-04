package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

class ChannelClusterIO(val b: GlobalConfig) extends Bundle {
  val channelIn       = Vec(b.top.ballMemChannelNum, Flipped(new ChannelIO(b)))
  val channelOut      = Vec(b.top.ballMemChannelNum, new ChannelIO(b))
  val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
  val freeChannelResp = Decoupled(new FreeChannelResp(b))
}

@instantiable
class ChannelCluster(val b: GlobalConfig) extends Module {
  @public
  val io = IO(new ChannelClusterIO(b))

  val channels = (0 until b.top.ballMemChannelNum).map(_ => Instantiate(new Channel(b)))

  // Connect channels
  for (i <- 0 until b.top.ballMemChannelNum) {
    channels(i).io.in <> io.channelIn(i)
    channels(i).io.out <> io.channelOut(i)
    channels(i).io.peakChannelReq.valid  := io.peakChannelReq.valid
    channels(i).io.peakChannelReq.bits   := io.peakChannelReq.bits
    channels(i).io.freeChannelResp.ready := true.B
  }

  val freeChannels      = VecInit(channels.map(ch => ch.io.freeChannelResp.bits.is_free))
  val freeChannelCount  = PopCount(freeChannels)
  val hasEnoughChannels = freeChannelCount >= io.peakChannelReq.bits.needed_channel_num

  val selectedChannelIds = Wire(Vec(b.top.ballMemChannelNum, UInt(log2Up(b.top.ballMemChannelNum).W)))
  val selectedMask       = Wire(Vec(b.top.ballMemChannelNum, Bool()))

  val masks = Wire(Vec(b.top.ballMemChannelNum + 1, Vec(b.top.ballMemChannelNum, Bool())))
  masks(0) := freeChannels

  for (i <- 0 until b.top.ballMemChannelNum) {
    val maskVec = masks(i)
    val selOH   = PriorityEncoderOH(maskVec.asUInt)
    val selIdx  = PriorityEncoder(maskVec.asUInt)
    selectedMask(i)       := maskVec.asUInt.orR && (i.U < io.peakChannelReq.bits.needed_channel_num)
    selectedChannelIds(i) := Mux(selectedMask(i), selIdx, 0.U)
    if (i < b.top.ballMemChannelNum) {
      masks(i + 1) := VecInit(maskVec.zip(selOH.asBools).map { case (free, sel) => free && !sel })
    }
  }

  // -----------------------------
  // Convert selectedChannelIds to "whether each channel is selected".
  // -----------------------------
  val selectedPerChan = Wire(Vec(b.top.ballMemChannelNum, Bool()))
  for (ch <- 0 until b.top.ballMemChannelNum) {
    // If the ch appears in the list of selected ids (and the slot is valid), it is considered selected.
    selectedPerChan(ch) := (0 until b.top.ballMemChannelNum).map { k =>
      selectedMask(k) && (selectedChannelIds(k) === ch.U)
    }.reduce(_ || _)
  }

 // Allocation is only truly "distributed" when the upstream handshake is successful and sufficient resources are available.
  val doAlloc = io.peakChannelReq.valid && io.freeChannelResp.ready && hasEnoughChannels

// Only raise the valid flag for the selected channel to avoid locking all channels simultaneously by firing.
  for (i <- 0 until b.top.ballMemChannelNum) {
    channels(i).io.peakChannelReq.valid := doAlloc && selectedPerChan(i)
  }

  io.freeChannelResp.valid            := io.peakChannelReq.valid
  io.freeChannelResp.bits.is_free     := hasEnoughChannels
  io.freeChannelResp.bits.channel_ids := selectedChannelIds
  io.freeChannelResp.bits.channel_num := io.peakChannelReq.bits.needed_channel_num
  io.peakChannelReq.ready             := io.freeChannelResp.ready
}
