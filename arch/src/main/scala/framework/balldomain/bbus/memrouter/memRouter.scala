package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

class PeakChannelReq(val b: GlobalConfig) extends Bundle {
  val needed_channel_num = UInt(log2Up(b.ballDomain.ballIdMappings.map(_.inBW).sum).W)
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
// Step1: Generate read request (probe only)
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

// 默认不写映射/不清理（多写口：全部置空）
  for (w <- 0 until bbusProducerChannels) {
    channelMappingTable.io.write(w).valid      := false.B
    channelMappingTable.io.write(w).bits.idx   := 0.U
    channelMappingTable.io.write(w).bits.outCh := 0.U

    channelMappingTable.io.invalidate(w).valid := false.B
    channelMappingTable.io.invalidate(w).bits  := 0.U
  }

// Step4: Set the channel mapping table (batch write)
  when(readReqGen.io.read_req_o.fire) {
    val matchVec = VecInit((0 until totalReadChannels).map { i =>
      io.bankRead_i(i).ball_id === readReqGen.io.read_req_o.bits.ball_id &&
      io.bankRead_i(i).bank_id === readReqGen.io.read_req_o.bits.bank_id &&
      io.bankRead_i(i).io.req.valid
    })

    val matchCount = PopCount(matchVec)
    val channelNum = readReqGen.io.read_req_o.bits.channel_num

    val portLimit     = bbusProducerChannels.U
    val wantCount     = Mux(matchCount > channelNum, channelNum, matchCount)
    val dispatchCount = Mux(wantCount > portLimit, portLimit, wantCount)
    
    // Assert: outCh used in this dispatch must be unique (no two inputs drive one output)
    for (p <- 0 until bbusProducerChannels) {
      for (q <- p + 1 until bbusProducerChannels) {
        when((p.U < dispatchCount) && (q.U < dispatchCount)) {
          assert(
            dispatchChannels(p) =/= dispatchChannels(q),
            "MemRouter: duplicate outCh in dispatchChannels within dispatchCount"
          )
        }
      }
    }
    for (i <- 0 until totalReadChannels) {
      when(matchVec(i)) {
        val priorMatchCount = PopCount(matchVec.take(i))
        when(priorMatchCount < dispatchCount) {
          // 第 k 个匹配输入 -> 第 k 个写口
          channelMappingTable.io.write(priorMatchCount).valid      := true.B
          channelMappingTable.io.write(priorMatchCount).bits.idx   := i.U
          channelMappingTable.io.write(priorMatchCount).bits.outCh := dispatchChannels(priorMatchCount)
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

  for (i <- 0 until totalReadChannels) {
    io.bankRead_i(i).io.req.ready  := false.B
    io.bankRead_i(i).io.resp.valid := false.B
    io.bankRead_i(i).io.resp.bits  := DontCare
  }

// Step5: Dispatch the request to the channels + entry-level invalidation
  for (i <- 0 until totalReadChannels) {
    when(channelMappingTable.io.routeValid(i)) {
      val outCh = channelMappingTable.io.routeMap(i)

      io.bankRead_o(outCh).io.req.valid := io.bankRead_i(i).io.req.valid
      io.bankRead_o(outCh).io.req.bits  := io.bankRead_i(i).io.req.bits
      io.bankRead_o(outCh).bank_id      := io.bankRead_i(i).bank_id
      io.bankRead_o(outCh).ball_id      := io.bankRead_i(i).ball_id
      io.bankRead_o(outCh).rob_id       := io.bankRead_i(i).rob_id

      io.bankRead_i(i).io.req.ready := io.bankRead_o(outCh).io.req.ready

      // 当该输入真正 fire（握手成功）后，清掉它的映射条目
      val didFire = io.bankRead_i(i).io.req.fire
      for (o <- 0 until bbusProducerChannels) {
        when(outCh === o.U) {
          channelMappingTable.io.invalidate(o).valid := didFire
          channelMappingTable.io.invalidate(o).bits  := i.U
        }
      }
    }
  }



//---------------------------------------------------------------------------
// Write: 仍未路由（保持你当前的默认行为）
//---------------------------------------------------------------------------
  for (i <- 0 until totalWriteChannels) {
    io.bankWrite_i(i).io.req.ready  := true.B
    io.bankWrite_i(i).io.resp.valid := false.B
    io.bankWrite_i(i).io.resp.bits  := DontCare
  }
}