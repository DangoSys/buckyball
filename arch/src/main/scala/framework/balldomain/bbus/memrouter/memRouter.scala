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
  val channel_ids = Vec(b.top.ballMemChannelNum, UInt(log2Up(b.top.ballMemChannelNum).W))
  val channel_num = UInt(log2Up(b.top.ballMemChannelNum + 1).W)
}

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val numBalls             = b.ballDomain.ballNum
  val bbusProducerChannels = b.top.ballMemChannelNum
  val bbusConsumerChannels = b.top.memBallChannelNum
  val totalReadChannels    = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels   = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val maxPerChannelWidth   = b.ballDomain.ballIdMappings.flatMap(m => Seq(m.inBW, m.outBW)).max

  @public
  val io = IO(new Bundle {
    val bankRead_i = Vec(totalReadChannels, new BankRead(b))
    val bankRead_o = Vec(bbusProducerChannels, Flipped(new BankRead(b)))

    val bankWrite_i = Vec(totalWriteChannels, new BankWrite(b))
    val bankWrite_o = Vec(bbusProducerChannels, Flipped(new BankWrite(b)))

    val peakChannelReq  = Decoupled(new PeakChannelReq(b))
    val freeChannelResp = Flipped(Decoupled(new FreeChannelResp(b)))
  })

//---------------------------------------------------------------------------
// Read
//---------------------------------------------------------------------------
// Step1: Generate read request 
  val readReqGen:          Instance[ReadReqGen]          = Instantiate(new ReadReqGen(b))
  val channelMappingTable: Instance[ChannelMappingTable] = Instantiate(new ChannelMappingTable(b))

  for (i <- 0 until totalReadChannels) {
    readReqGen.io.bank_read_i(i).valid   := io.bankRead_i(i).io.req.valid
    readReqGen.io.bank_read_i(i).ball_id := io.bankRead_i(i).ball_id
    readReqGen.io.bank_read_i(i).bank_id := io.bankRead_i(i).bank_id
    readReqGen.io.bank_read_i(i).rob_id  := io.bankRead_i(i).rob_id
  }

// Step2: Peek if there are enough free channels
  val hasReadReq = readReqGen.io.read_req_o.valid
  io.peakChannelReq.valid                   := hasReadReq
  io.peakChannelReq.bits.needed_channel_num := readReqGen.io.read_req_o.bits.channel_num
  io.peakChannelReq.bits.bank_id            := readReqGen.io.read_req_o.bits.bank_id
  io.peakChannelReq.bits.rob_id             := readReqGen.io.read_req_o.bits.rob_id

// Step3: accept the request only when enough channels
  val isChannelFree = io.freeChannelResp.valid && io.freeChannelResp.bits.is_free
  io.freeChannelResp.ready       := true.B
  val dispatchChannels           = io.freeChannelResp.bits.channel_ids
  readReqGen.io.read_req_o.ready := isChannelFree

// 默认不写映射
  channelMappingTable.io.write.valid      := false.B
  channelMappingTable.io.write.bits.idx   := 0.U
  channelMappingTable.io.write.bits.outCh := 0.U

// Step4: Set the channel mapping table
  when(readReqGen.io.read_req_o.fire) {
    val matchVec = VecInit((0 until totalReadChannels).map { i =>
      io.bankRead_i(i).ball_id === readReqGen.io.read_req_o.bits.ball_id &&
      io.bankRead_i(i).bank_id === readReqGen.io.read_req_o.bits.bank_id &&
      io.bankRead_i(i).io.req.valid
    })
    val matchCount    = PopCount(matchVec)
    val channelNum    = readReqGen.io.read_req_o.bits.channel_num
    val dispatchCount = Mux(matchCount > channelNum, channelNum, matchCount)

    for (i <- 0 until totalReadChannels) {
      when(matchVec(i)) {
        val priorMatchCount = PopCount(matchVec.take(i))
        when(priorMatchCount < dispatchCount) {
          channelMappingTable.io.write.valid      := true.B
          channelMappingTable.io.write.bits.idx   := i.U
          channelMappingTable.io.write.bits.outCh := dispatchChannels(priorMatchCount)
        }
      }
    }
  }

//-------------------- 默认值：输出端全空闲，输入端默认不 ready（避免吞请求） --------------------
  for (o <- 0 until bbusProducerChannels) {
    io.bankRead_o(o).io.req.valid  := false.B
    io.bankRead_o(o).io.req.bits   := DontCare
    io.bankRead_o(o).io.resp.ready := true.B
    io.bankRead_o(o).bank_id       := DontCare
    io.bankRead_o(o).ball_id       := DontCare
    io.bankRead_o(o).rob_id        := DontCare

    io.bankWrite_o(o).io.req.valid  := false.B
    io.bankWrite_o(o).io.req.bits   := DontCare
    io.bankWrite_o(o).io.resp.ready := true.B
    io.bankWrite_o(o).bank_id       := DontCare
    io.bankWrite_o(o).ball_id       := DontCare
    io.bankWrite_o(o).rob_id        := DontCare
  }

  // 输入端：默认不接受请求（等到被映射到 outCh 再根据 outCh.ready 回推）
  for (i <- 0 until totalReadChannels) {
    io.bankRead_i(i).io.req.ready  := false.B
    io.bankRead_i(i).io.resp.valid := false.B
    io.bankRead_i(i).io.resp.bits  := DontCare
  }

// Step5: Dispatch the request to the channels（显式连 req，并回推 ready）
  for (i <- 0 until totalReadChannels) {
    when(channelMappingTable.io.routeValid(i)) {
      val outCh = channelMappingTable.io.routeMap(i)

      // drive output req from selected input
      io.bankRead_o(outCh).io.req.valid := io.bankRead_i(i).io.req.valid
      io.bankRead_o(outCh).io.req.bits  := io.bankRead_i(i).io.req.bits
      io.bankRead_o(outCh).bank_id      := io.bankRead_i(i).bank_id
      io.bankRead_o(outCh).ball_id      := io.bankRead_i(i).ball_id
      io.bankRead_o(outCh).rob_id       := io.bankRead_i(i).rob_id

      // backpressure back to input
      io.bankRead_i(i).io.req.ready := io.bankRead_o(outCh).io.req.ready
    }
  }

//---------------------------------------------------------------------------
// Write: 你原代码这里还没理顺（读写索引空间不一致），我保持不动/默认不派发
//---------------------------------------------------------------------------

  for (i <- 0 until totalWriteChannels) {
    io.bankWrite_i(i).io.req.ready  := true.B
    io.bankWrite_i(i).io.resp.valid := false.B
    io.bankWrite_i(i).io.resp.bits  := DontCare
  }
}