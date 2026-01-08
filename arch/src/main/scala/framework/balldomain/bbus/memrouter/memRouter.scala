package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

class PeakChannelReq(val b: GlobalConfig) extends Bundle {
  val needed_channel_num = UInt(log2Up(b.top.ballMemChannelNum + 1).W)
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
  val bbusProducerChannels = b.top.ballMemChannelNum
  val totalReadChannels    = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalWriteChannels   = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
    val bankRead_i  = Vec(totalReadChannels, new BankRead(b))
    val bankRead_o  = Vec(bbusProducerChannels, Flipped(new BankRead(b)))

    val bankWrite_i = Vec(totalWriteChannels, new BankWrite(b))
    val bankWrite_o = Vec(bbusProducerChannels, Flipped(new BankWrite(b)))

    val peakChannelReq  = Decoupled(new PeakChannelReq(b))
    val freeChannelResp = Flipped(Decoupled(new FreeChannelResp(b)))
  })

  // ---------------------------------------------------------------------------
  // Step1: probe -> ReadReqGen / WriteReqGen
  // ---------------------------------------------------------------------------
  val readReqGen:  Instance[ReadReqGen]  = Instantiate(new ReadReqGen(b))
  val writeReqGen: Instance[WriteReqGen] = Instantiate(new WriteReqGen(b))

  val readMap:  Instance[ChannelMappingTable] = Instantiate(new ChannelMappingTable(b, totalReadChannels))
  val writeMap: Instance[ChannelMappingTable] = Instantiate(new ChannelMappingTable(b, totalWriteChannels))

  for (i <- 0 until totalReadChannels) {
    readReqGen.io.bank_read_i(i).valid   := io.bankRead_i(i).io.req.valid
    readReqGen.io.bank_read_i(i).ball_id := io.bankRead_i(i).ball_id
    readReqGen.io.bank_read_i(i).bank_id := io.bankRead_i(i).bank_id
    readReqGen.io.bank_read_i(i).rob_id  := io.bankRead_i(i).rob_id
  }
  for (i <- 0 until totalWriteChannels) {
    writeReqGen.io.bank_write_i(i).valid   := io.bankWrite_i(i).io.req.valid
    writeReqGen.io.bank_write_i(i).ball_id := io.bankWrite_i(i).ball_id
    writeReqGen.io.bank_write_i(i).bank_id := io.bankWrite_i(i).bank_id
    writeReqGen.io.bank_write_i(i).rob_id  := io.bankWrite_i(i).rob_id
  }

  // ---------------------------------------------------------------------------
  // Step2: one allocator, RR choose read/write group
  // ---------------------------------------------------------------------------
  val hasReadReq  = readReqGen.io.read_req_o.valid
  val hasWriteReq = writeReqGen.io.write_req_o.valid

  val rrLastWasRead = RegInit(true.B) // 用于读写公平：true 表示上次发的是读

  val chooseRead = WireDefault(false.B)
  when(hasReadReq && !hasWriteReq) {
    chooseRead := true.B
  }.elsewhen(!hasReadReq && hasWriteReq) {
    chooseRead := false.B
  }.elsewhen(hasReadReq && hasWriteReq) {
    chooseRead := !rrLastWasRead
  }

  val chooseWrite = hasReadReq && hasWriteReq && !chooseRead || (!hasReadReq && hasWriteReq)

  io.peakChannelReq.valid                   := hasReadReq || hasWriteReq
  io.peakChannelReq.bits.needed_channel_num := Mux(chooseRead, readReqGen.io.read_req_o.bits.channel_num, writeReqGen.io.write_req_o.bits.channel_num)
  io.peakChannelReq.bits.bank_id            := Mux(chooseRead, readReqGen.io.read_req_o.bits.bank_id, writeReqGen.io.write_req_o.bits.bank_id)
  io.peakChannelReq.bits.rob_id             := Mux(chooseRead, readReqGen.io.read_req_o.bits.rob_id, writeReqGen.io.write_req_o.bits.rob_id)

  // allocator 这里当“组合查询”用
  io.freeChannelResp.ready := true.B
  val isChannelFree = io.freeChannelResp.valid && io.freeChannelResp.bits.is_free && io.peakChannelReq.valid
  val dispatchChannels = io.freeChannelResp.bits.channel_ids

  readReqGen.io.read_req_o.ready   := isChannelFree && chooseRead
  writeReqGen.io.write_req_o.ready := isChannelFree && chooseWrite

  val didDispatchRead  = readReqGen.io.read_req_o.fire
  val didDispatchWrite = writeReqGen.io.write_req_o.fire

  when(didDispatchRead)  { rrLastWasRead := true.B }
  when(didDispatchWrite) { rrLastWasRead := false.B }

  // ---------------------------------------------------------------------------
  // defaults: mapping write/invalidate off
  // ---------------------------------------------------------------------------
  for (w <- 0 until bbusProducerChannels) {
    readMap.io.write(w).valid      := false.B
    readMap.io.write(w).bits.idx   := 0.U
    readMap.io.write(w).bits.outCh := 0.U
    readMap.io.invalidate(w).valid := false.B
    readMap.io.invalidate(w).bits  := 0.U

    writeMap.io.write(w).valid      := false.B
    writeMap.io.write(w).bits.idx   := 0.U
    writeMap.io.write(w).bits.outCh := 0.U
    writeMap.io.invalidate(w).valid := false.B
    writeMap.io.invalidate(w).bits  := 0.U
  }

  // ---------------------------------------------------------------------------
  // Step4: batch write mapping for chosen side
  // ---------------------------------------------------------------------------
  when(didDispatchRead) {
    val matchVec = VecInit((0 until totalReadChannels).map { i =>
      io.bankRead_i(i).ball_id === readReqGen.io.read_req_o.bits.ball_id &&
      io.bankRead_i(i).bank_id === readReqGen.io.read_req_o.bits.bank_id &&
      io.bankRead_i(i).io.req.valid
    })

    val matchCount   = PopCount(matchVec)
    val channelNum   = readReqGen.io.read_req_o.bits.channel_num
    val portLimit    = bbusProducerChannels.U
    val wantCount    = Mux(matchCount > channelNum, channelNum, matchCount)
    val dispatchCount = Mux(wantCount > portLimit, portLimit, wantCount)

    for (p <- 0 until bbusProducerChannels) {
      for (q <- p + 1 until bbusProducerChannels) {
        when((p.U < dispatchCount) && (q.U < dispatchCount)) {
          assert(
            dispatchChannels(p) =/= dispatchChannels(q),
            "MemRouter(Read): duplicate outCh in dispatchChannels within dispatchCount"
          )
        }
      }
    }

    for (i <- 0 until totalReadChannels) {
      when(matchVec(i)) {
        val priorMatchCount = PopCount(matchVec.take(i))
        when(priorMatchCount < dispatchCount) {
          readMap.io.write(priorMatchCount).valid      := true.B
          readMap.io.write(priorMatchCount).bits.idx   := i.U
          readMap.io.write(priorMatchCount).bits.outCh := dispatchChannels(priorMatchCount)
        }
      }
    }
  }

  when(didDispatchWrite) {
    val matchVec = VecInit((0 until totalWriteChannels).map { i =>
      io.bankWrite_i(i).ball_id === writeReqGen.io.write_req_o.bits.ball_id &&
      io.bankWrite_i(i).bank_id === writeReqGen.io.write_req_o.bits.bank_id &&
      io.bankWrite_i(i).io.req.valid
    })

    val matchCount   = PopCount(matchVec)
    val channelNum   = writeReqGen.io.write_req_o.bits.channel_num
    val portLimit    = bbusProducerChannels.U
    val wantCount    = Mux(matchCount > channelNum, channelNum, matchCount)
    val dispatchCount = Mux(wantCount > portLimit, portLimit, wantCount)

    for (p <- 0 until bbusProducerChannels) {
      for (q <- p + 1 until bbusProducerChannels) {
        when((p.U < dispatchCount) && (q.U < dispatchCount)) {
          assert(
            dispatchChannels(p) =/= dispatchChannels(q),
            "MemRouter(Write): duplicate outCh in dispatchChannels within dispatchCount"
          )
        }
      }
    }

    for (i <- 0 until totalWriteChannels) {
      when(matchVec(i)) {
        val priorMatchCount = PopCount(matchVec.take(i))
        when(priorMatchCount < dispatchCount) {
          writeMap.io.write(priorMatchCount).valid      := true.B
          writeMap.io.write(priorMatchCount).bits.idx   := i.U
          writeMap.io.write(priorMatchCount).bits.outCh := dispatchChannels(priorMatchCount)
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // defaults: out ports idle; in ports backpressured; resp invalid
  // ---------------------------------------------------------------------------
  for (o <- 0 until bbusProducerChannels) {
    io.bankRead_o(o).io.req.valid  := false.B
    io.bankRead_o(o).io.req.bits   := DontCare
    io.bankRead_o(o).io.resp.ready := false.B
    io.bankRead_o(o).bank_id       := DontCare
    io.bankRead_o(o).ball_id       := DontCare
    io.bankRead_o(o).rob_id        := DontCare

    io.bankWrite_o(o).io.req.valid  := false.B
    io.bankWrite_o(o).io.req.bits   := DontCare
    io.bankWrite_o(o).io.resp.ready := false.B
    io.bankWrite_o(o).bank_id       := DontCare
    io.bankWrite_o(o).ball_id       := DontCare
    io.bankWrite_o(o).rob_id        := DontCare
  }

  for (i <- 0 until totalReadChannels) {
    io.bankRead_i(i).io.req.ready  := false.B
    io.bankRead_i(i).io.resp.valid := false.B
    io.bankRead_i(i).io.resp.bits  := DontCare
  }
  for (i <- 0 until totalWriteChannels) {
    io.bankWrite_i(i).io.req.ready  := false.B
    io.bankWrite_i(i).io.resp.valid := false.B
    io.bankWrite_i(i).io.resp.bits  := DontCare
  }

  // ---------------------------------------------------------------------------
  // Step5: route req/resp by outCh, invalidate on resp.fire
  // ---------------------------------------------------------------------------
  for (o <- 0 until bbusProducerChannels) {
    val readIdxOH = VecInit((0 until totalReadChannels).map { i =>
      readMap.io.routeValid(i) && (readMap.io.routeMap(i) === o.U)
    })
    val writeIdxOH = VecInit((0 until totalWriteChannels).map { i =>
      writeMap.io.routeValid(i) && (writeMap.io.routeMap(i) === o.U)
    })

    val hasRead  = readIdxOH.asUInt.orR
    val hasWrite = writeIdxOH.asUInt.orR

    assert(!(hasRead && hasWrite), "MemRouter: one outCh is mapped by both read and write")

    when(hasRead) {
      val ridx = PriorityEncoder(readIdxOH.asUInt)

      io.bankRead_o(o).io.req.valid := io.bankRead_i(ridx).io.req.valid
      io.bankRead_o(o).io.req.bits  := io.bankRead_i(ridx).io.req.bits
      io.bankRead_o(o).bank_id      := io.bankRead_i(ridx).bank_id
      io.bankRead_o(o).ball_id      := io.bankRead_i(ridx).ball_id
      io.bankRead_o(o).rob_id       := io.bankRead_i(ridx).rob_id

      io.bankRead_i(ridx).io.req.ready := io.bankRead_o(o).io.req.ready

      io.bankRead_i(ridx).io.resp.valid := io.bankRead_o(o).io.resp.valid
      io.bankRead_i(ridx).io.resp.bits  := io.bankRead_o(o).io.resp.bits
      io.bankRead_o(o).io.resp.ready    := io.bankRead_i(ridx).io.resp.ready

      when(io.bankRead_o(o).io.resp.fire) {
        readMap.io.invalidate(o).valid := true.B
        readMap.io.invalidate(o).bits  := ridx
      }
    }.elsewhen(hasWrite) {
      val widx = PriorityEncoder(writeIdxOH.asUInt)

      io.bankWrite_o(o).io.req.valid := io.bankWrite_i(widx).io.req.valid
      io.bankWrite_o(o).io.req.bits  := io.bankWrite_i(widx).io.req.bits
      io.bankWrite_o(o).bank_id      := io.bankWrite_i(widx).bank_id
      io.bankWrite_o(o).ball_id      := io.bankWrite_i(widx).ball_id
      io.bankWrite_o(o).rob_id       := io.bankWrite_i(widx).rob_id

      io.bankWrite_i(widx).io.req.ready := io.bankWrite_o(o).io.req.ready

      io.bankWrite_i(widx).io.resp.valid := io.bankWrite_o(o).io.resp.valid
      io.bankWrite_i(widx).io.resp.bits  := io.bankWrite_o(o).io.resp.bits
      io.bankWrite_o(o).io.resp.ready    := io.bankWrite_i(widx).io.resp.ready

      when(io.bankWrite_o(o).io.resp.fire) {
        writeMap.io.invalidate(o).valid := true.B
        writeMap.io.invalidate(o).bits  := widx
      }
    }
  }
}