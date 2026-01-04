package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

class PeakChannelReq(val b: GlobalConfig) extends Bundle {
  val needed_channel_num = UInt(log2Up(b.ballDomain.ballIdMappings.map(_.inBW).sum + 1).W)
  val bank_id            = UInt(log2Up(b.memDomain.bankNum).W)
  val rob_id             = UInt(log2Up(b.frontend.rob_entries).W)
}

class FreeChannelResp(val b: GlobalConfig) extends Bundle {
  val is_free     = Bool()
  val channel_ids = Vec(b.top.ballMemChannelProducer, UInt(log2Up(b.top.ballMemChannelProducer).W))
  val channel_num = UInt(log2Up(b.top.ballMemChannelProducer + 1).W)
}

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val numBalls             = b.ballDomain.ballNum
  val bbusProducerChannels = b.top.ballMemChannelProducer
  val bbusConsumerChannels = b.top.ballMemChannelConsumer
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

//---------------------------------------------------------------------------
// Read
//---------------------------------------------------------------------------
// Step1: Generate read request
  val readReqGen:          Instance[ReadReqGen]          = Instantiate(new ReadReqGen(b))
  val channelMappingTable: Instance[ChannelMappingTable] = Instantiate(new ChannelMappingTable(b))
  readReqGen.io.bank_read_i <> io.bankRead_i
// Step2: Peek if there are enough free channels
  val hasReadReq = readReqGen.io.read_req_o.valid
  io.peakChannelReq.valid                   := hasReadReq
  io.peakChannelReq.bits.needed_channel_num := readReqGen.io.read_req_o.bits.channel_num
  io.peakChannelReq.bits.bank_id            := readReqGen.io.read_req_o.bits.bank_id
  io.peakChannelReq.bits.rob_id             := readReqGen.io.read_req_o.bits.rob_id
// Step3/1: if there are enough free channels, accept the request
  val isChannelFree = io.freeChannelResp.valid && io.freeChannelResp.bits.is_free
  io.freeChannelResp.ready := true.B
  val dispatchChannels = io.freeChannelResp.bits.channel_ids
  readReqGen.io.read_req_o.ready := isChannelFree
// Step3/2: if there are not enough free channels, continue to peek until there are enough free channels

// Step4: Set the channel mapping table
  when(readReqGen.io.read_req_o.fire) {
    var matchedCnt = 0.U
    for (i <- 0 until totalReadChannels) {
      val isMatch = io.bankRead_i(i).ball_id === readReqGen.io.read_req_o.bits.ball_id &&
        io.bankRead_i(i).bank_id === readReqGen.io.read_req_o.bits.bank_id && io.bankRead_i(i).io.req.valid
      when(isMatch && matchedCnt < readReqGen.io.read_req_o.bits.channel_num) {
        channelMappingTable.io.write.valid      := true.B
        channelMappingTable.io.write.bits.idx   := i.U
        channelMappingTable.io.write.bits.outCh := dispatchChannels(matchedCnt)
        matchedCnt = matchedCnt + 1.U
      }
    }
  }
// Step5: Dispatch the request to the channels
  for (i <- 0 until totalReadChannels) {
    when(channelMappingTable.io.routeValid(i)) {
      val outCh = channelMappingTable.io.routeMap(i)
      io.bankRead_o(outCh).io.req <> io.bankRead_i(i).io.req
      io.bankRead_o(outCh).ball_id := io.bankRead_i(i).ball_id
      io.bankRead_o(outCh).bank_id := io.bankRead_i(i).bank_id
      io.bankRead_o(outCh).rob_id  := io.bankRead_i(i).rob_id
    }
  }
}
