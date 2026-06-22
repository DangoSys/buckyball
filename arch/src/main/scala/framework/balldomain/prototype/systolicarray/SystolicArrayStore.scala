package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.SramWriteIO
import framework.top.GlobalConfig
import framework.balldomain.prototype.systolicarray.configs.SystolicBallParam

class ctrl_st_req(b: GlobalConfig) extends Bundle {
  val wr_bank      = UInt(log2Up(b.memDomain.bankNum).W)
  val wr_bank_addr = UInt(log2Up(b.memDomain.bankEntries).W)
  val iter         = UInt(b.frontend.iter_len.W)
  val config       = UInt(64.W)
}

class BankWriteEntry(b: GlobalConfig) extends Bundle {
  val addr  = UInt(log2Ceil(b.memDomain.bankEntries).W)
  val data  = UInt(b.memDomain.bankWidth.W)
  val mask  = Vec(b.memDomain.bankMaskLen, Bool())
  val wmode = Bool()
}

@instantiable
class SystolicArrayStore(val b: GlobalConfig) extends Module {
  val config   = SystolicBallParam()
  val InputNum = config.lane
  val accWidth = config.outputWidth
  val accModeLsb = 1
  val accModeWidth = 2
  val directMode = 0.U(accModeWidth.W)
  val firstMode = 1.U(accModeWidth.W)
  val midMode = 2.U(accModeWidth.W)
  val lastMode = 3.U(accModeWidth.W)

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  val outBW       = ballMapping.outBW

  require(InputNum % outBW == 0)
  require(
    InputNum / outBW * accWidth <= b.memDomain.bankWidth,
    s"SystolicArrayBall outBW=$outBW is too small for lane=$InputNum (need at least ${InputNum * accWidth / b.memDomain.bankWidth} write ports)"
  )

  @public
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new ctrl_st_req(b)))
    val ex_st_i   = Flipped(Decoupled(new ex_st_req(b)))
    val bankWrite = Vec(outBW, Flipped(new SramWriteIO(b)))
    val wr_bank_o = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  val wr_bank             = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank_addr        = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val iter_counter        = RegInit(0.U(b.frontend.iter_len.W))
  val expectedRows        = RegInit(0.U(b.frontend.iter_len.W))
  val configReg           = RegInit(0.U(64.W))
  val accRowsSeen         = RegInit(0.U(b.frontend.iter_len.W))
  val flushRowCounter     = RegInit(0.U(b.frontend.iter_len.W))
  val accBuf = RegInit(
    VecInit(Seq.fill(InputNum)(VecInit(Seq.fill(InputNum)(0.U(accWidth.W)))))
  )
  val idle :: busy :: flushing :: Nil = Enum(3)
  val state               = RegInit(idle)

  val writeQueues = VecInit(Seq.fill(outBW)(Module(new Queue(new BankWriteEntry(b), 16)).io))
  val allQueuesReady = writeQueues.map(_.enq.ready).reduce(_ && _)
  val allQueuesEmpty = writeQueues.map(q => !q.deq.valid).reduce(_ && _)
  val accMode = configReg(accModeLsb + accModeWidth - 1, accModeLsb)

  private def clearAccBuf(): Unit = {
    for (row <- 0 until InputNum) {
      for (col <- 0 until InputNum) {
        accBuf(row)(col) := 0.U
      }
    }
  }

  private def enqueueRow(rowData: Seq[UInt], rowIdx: UInt): Unit = {
    for (i <- 0 until outBW) {
      val elementsPerChannel = InputNum / outBW
      val startIdx           = i * elementsPerChannel
      val endIdx             = startIdx + elementsPerChannel - 1

      val entry = Wire(new BankWriteEntry(b))
      entry.addr  := wr_bank_addr + rowIdx
      entry.data  := Cat(rowData.slice(startIdx, endIdx + 1).reverse)
      entry.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
      entry.wmode := true.B

      writeQueues(i).enq.valid := true.B
      writeQueues(i).enq.bits  := entry
    }
  }

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    val iterClamped = Mux(io.ctrl_st_i.bits.iter > InputNum.U, InputNum.U, io.ctrl_st_i.bits.iter)
    expectedRows := iterClamped
    iter_counter := 0.U
    accRowsSeen  := 0.U
    flushRowCounter := 0.U
    configReg    := io.ctrl_st_i.bits.config
    when(io.ctrl_st_i.bits.config(accModeLsb + accModeWidth - 1, accModeLsb) === directMode ||
      io.ctrl_st_i.bits.config(accModeLsb + accModeWidth - 1, accModeLsb) === firstMode) {
      clearAccBuf()
    }
    state        := busy
  }

// -----------------------------------------------------------------------------
// Accept computation results from EX unit and push to write queues
// -----------------------------------------------------------------------------
  io.ex_st_i.ready := state === busy && allQueuesReady

  for (i <- 0 until outBW) {
    writeQueues(i).enq.valid := false.B
    writeQueues(i).enq.bits  := DontCare
  }

  when(io.ex_st_i.fire) {
    val rowIdx = accRowsSeen
    when(accMode === directMode) {
      enqueueRow(io.ex_st_i.bits.result, iter_counter)
      iter_counter := iter_counter + 1.U
    }.otherwise {
      for (col <- 0 until InputNum) {
        accBuf(rowIdx)(col) := accBuf(rowIdx)(col) + io.ex_st_i.bits.result(col)
      }
      accRowsSeen := accRowsSeen + 1.U
      when(accMode === lastMode && accRowsSeen + 1.U >= expectedRows) {
        flushRowCounter := 0.U
        iter_counter := 0.U
        state := flushing
      }
    }
  }

// -----------------------------------------------------------------------------
// Drain write queues to bankWrite interface
// -----------------------------------------------------------------------------
  io.bankWrite.foreach { acc =>
    acc.req.valid      := false.B
    acc.req.bits.addr  := 0.U
    acc.req.bits.data  := 0.U(b.memDomain.bankWidth.W)
    acc.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    acc.req.bits.wmode := false.B
    acc.resp.ready     := false.B
  }

  for (i <- 0 until outBW) {
    writeQueues(i).deq.ready := false.B

    when(writeQueues(i).deq.valid) {
      io.bankWrite(i).req.valid      := true.B
      io.bankWrite(i).req.bits.addr  := writeQueues(i).deq.bits.addr
      io.bankWrite(i).req.bits.data  := writeQueues(i).deq.bits.data
      io.bankWrite(i).req.bits.mask  := writeQueues(i).deq.bits.mask
      io.bankWrite(i).req.bits.wmode := writeQueues(i).deq.bits.wmode
      writeQueues(i).deq.ready       := io.bankWrite(i).req.ready
    }
  }

  when(state === flushing && allQueuesReady && flushRowCounter < expectedRows) {
    enqueueRow(accBuf(flushRowCounter), flushRowCounter)
    flushRowCounter := flushRowCounter + 1.U
  }

  // Output wr_bank for bank_id setting
  io.wr_bank_o := wr_bank

// -----------------------------------------------------------------------------
// Reset iter counter, commit cmdResp, return to idle state
// -----------------------------------------------------------------------------
  val directDone = state === busy && accMode === directMode && iter_counter >= expectedRows
  val accBufferedDone =
    state === busy && accMode =/= directMode && accMode =/= lastMode && accRowsSeen >= expectedRows
  val flushDone =
    state === flushing && flushRowCounter >= expectedRows
  val allDataEnqueued = directDone || accBufferedDone || flushDone

  when(allDataEnqueued && allQueuesEmpty) {
    state                    := idle
    expectedRows             := 0.U
    accRowsSeen              := 0.U
    flushRowCounter          := 0.U
    iter_counter             := 0.U
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.commit := false.B
  }

  when(state === idle) {
    iter_counter := 0.U
    configReg    := 0.U
  }

}
