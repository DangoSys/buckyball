package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.im2col.configs.Im2colBallParam

@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  private val maxK        = Im2colBallParam().InputNum
  private val elemWidth   = Im2colBallParam().inputWidth
  private val bankWidth   = b.memDomain.bankWidth
  private val lanesPerBeat = 16
  private val laneWidth    = bankWidth / lanesPerBeat
  private val maxInCol     = 32
  private val maxInColWords = (maxInCol + lanesPerBeat - 1) / lanesPerBeat
  private val maxKernelElems = maxK * maxK

  require(laneWidth == elemWidth, s"[Im2col] laneWidth($laneWidth) must equal elemWidth($elemWidth)")

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))
  private val inBW  = mapping.inBW
  private val outBW = mapping.outBW

  @public val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  require(inBW >= 1, "[Im2col] inBW must be >= 1")
  require(outBW >= 1, "[Im2col] outBW must be >= 1")

  val idle :: preload_rows :: generate_window :: write_window :: load_next_row :: complete :: Nil = Enum(6)
  val state = RegInit(idle)

  private val robIdReg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val rBankReg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBankReg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))

  private val rBaseBeatReg = RegInit(0.U(32.W))
  private val wBaseBeatReg = RegInit(0.U(32.W))

  private val kRowReg     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kColReg     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val inRowReg    = RegInit(0.U(16.W))
  private val inColReg    = RegInit(0.U(16.W))
  private val startRowReg = RegInit(0.U(16.W))
  private val startColReg = RegInit(0.U(16.W))

  private val rowPtrReg = RegInit(0.U(16.W))
  private val colPtrReg = RegInit(0.U(16.W))

  private val lineBuffer = RegInit(VecInit(Seq.fill(maxK)(VecInit(Seq.fill(maxInColWords)(0.U(bankWidth.W))))))
  private val elemBuffer = RegInit(VecInit(Seq.fill(maxKernelElems)(0.U(elemWidth.W))))

  private val rowFifo = Module(new RowSlotFIFO(maxK))
  rowFifo.io.kRows   := kRowReg
  rowFifo.io.init    := false.B
  rowFifo.io.advance := false.B

  private val ldRowIdxReg = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val ldBeatIdxReg = RegInit(0.U(log2Ceil(maxInColWords + 1).W))
  private val ldOutstandingReg = RegInit(false.B)

  private val genElemIdxReg = RegInit(0.U(log2Ceil(maxKernelElems + 1).W))
  private val wrElemIdxReg = RegInit(0.U(log2Ceil(maxKernelElems + 1).W))
  private val packCntReg = RegInit(0.U(log2Ceil(lanesPerBeat + 1).W))
  private val packReg = RegInit(VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W))))

  private def ceilDiv(a: UInt, d: Int): UInt = (a + (d - 1).U) / d.U
  private val inColWords = ceilDiv(inColReg, lanesPerBeat)
  private val totalKernelEls = kRowReg * kColReg

  private val rowMax = inRowReg - kRowReg
  private val colMax = inColReg - kColReg
  private val rowEnd = rowPtrReg === (startRowReg + rowMax)
  private val colEnd = colPtrReg === (startColReg + colMax)
  private val isLastWindow = rowEnd && colEnd

  private def laneFromBeat(beat: UInt, lane: UInt): UInt = {
    val lanes = beat.asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))
    lanes(lane)
  }

  io.cmdReq.ready        := (state === idle)
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := robIdReg

  io.status.idle    := (state === idle)
  io.status.running := (state =/= idle) && (state =/= complete)

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := rBankReg
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := wBankReg
    io.bankWrite(i).rob_id            := robIdReg
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).group_id          := 0.U
  }

  io.bankWrite(0).io.resp.ready := true.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        robIdReg := io.cmdReq.bits.rob_id
        rBankReg := io.cmdReq.bits.cmd.op1_bank
        wBankReg := io.cmdReq.bits.cmd.wr_bank

        kColReg := io.cmdReq.bits.cmd.special(3, 0)
        kRowReg := io.cmdReq.bits.cmd.special(7, 4)
        inColReg := io.cmdReq.bits.cmd.special(12, 8)
        inRowReg := io.cmdReq.bits.cmd.special(22, 13)
        startColReg := io.cmdReq.bits.cmd.special(27, 23)
        startRowReg := io.cmdReq.bits.cmd.special(37, 28)

        rowPtrReg := io.cmdReq.bits.cmd.special(37, 28)
        colPtrReg := io.cmdReq.bits.cmd.special(27, 23)

        rBaseBeatReg := 0.U
        wBaseBeatReg := 0.U

        ldRowIdxReg := 0.U
        ldBeatIdxReg := 0.U
        ldOutstandingReg := false.B
        genElemIdxReg := 0.U
        wrElemIdxReg := 0.U
        packCntReg := 0.U

        rowFifo.io.init := true.B

        val cmdKCol = io.cmdReq.bits.cmd.special(3, 0)
        val cmdKRow = io.cmdReq.bits.cmd.special(7, 4)
        val cmdInCol = io.cmdReq.bits.cmd.special(12, 8)
        val cmdInRow = io.cmdReq.bits.cmd.special(22, 13)
        val invalidShape = (cmdKCol === 0.U) || (cmdKRow === 0.U) || (cmdInCol === 0.U) || (cmdInRow === 0.U) ||
          (cmdInCol < cmdKCol) || (cmdInRow < cmdKRow)

        when(invalidShape) {
          state := complete
        }.otherwise {
          state := preload_rows
        }
      }
    }

    is(preload_rows) {
      val doneRows = ldRowIdxReg === kRowReg
      val canIssue = !doneRows && !ldOutstandingReg && (ldBeatIdxReg < inColWords)
      val rowElem = rowPtrReg + ldRowIdxReg
      val reqAddr = rBaseBeatReg + rowElem * inColWords + ldBeatIdxReg

      io.bankRead(0).io.req.valid     := canIssue
      io.bankRead(0).io.req.bits.addr := reqAddr
      io.bankRead(0).io.resp.ready    := ldOutstandingReg

      when(io.bankRead(0).io.req.fire) {
        ldOutstandingReg := true.B
      }

      when(io.bankRead(0).io.resp.fire) {
        lineBuffer(ldRowIdxReg)(ldBeatIdxReg) := io.bankRead(0).io.resp.bits.data.asUInt
        ldOutstandingReg := false.B

        when(ldBeatIdxReg + 1.U === inColWords) {
          ldBeatIdxReg := 0.U
          ldRowIdxReg := ldRowIdxReg + 1.U
        }.otherwise {
          ldBeatIdxReg := ldBeatIdxReg + 1.U
        }
      }

      when(doneRows && !ldOutstandingReg) {
        genElemIdxReg := 0.U
        state := generate_window
      }
    }

    is(generate_window) {
      val startLane = colPtrReg % lanesPerBeat.U
      val safeKCol = Mux(kColReg === 0.U, 1.U, kColReg)

      val t = genElemIdxReg
      val kRowIdx = t / safeKCol
      val kColIdx = t % safeKCol

      val physicalSlot = RowSlotFIFO.logicalToPhysical(rowFifo.io.head, kRowIdx, kRowReg)
      val laneSum = startLane + kColIdx
      val beatIdx = laneSum / lanesPerBeat.U
      val laneIdx = laneSum % lanesPerBeat.U
      val beatWord = lineBuffer(physicalSlot)(beatIdx)

      elemBuffer(t) := laneFromBeat(beatWord, laneIdx)

      when(genElemIdxReg === (totalKernelEls - 1.U)) {
        wrElemIdxReg := 0.U
        state := write_window
      }.otherwise {
        genElemIdxReg := genElemIdxReg + 1.U
      }
    }

    is(write_window) {
      val windowDone = wrElemIdxReg === totalKernelEls
      val packFull = packCntReg === lanesPerBeat.U

      io.bankWrite(0).io.req.valid      := packFull
      io.bankWrite(0).io.req.bits.addr  := wBaseBeatReg
      io.bankWrite(0).io.req.bits.data  := Cat(packReg.reverse)
      io.bankWrite(0).io.req.bits.wmode := true.B
      io.bankWrite(0).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))

      when(!windowDone && !packFull) {
        packReg(packCntReg) := elemBuffer(wrElemIdxReg)
        packCntReg := packCntReg + 1.U
        wrElemIdxReg := wrElemIdxReg + 1.U
      }

      when(io.bankWrite(0).io.req.fire) {
        wBaseBeatReg := wBaseBeatReg + 1.U
        packCntReg := 0.U
      }

      when(windowDone) {
        when(packFull && !io.bankWrite(0).io.req.fire) {
          state := write_window
        }.otherwise {
          when(isLastWindow) {
            state := complete
          }.elsewhen(colEnd) {
            colPtrReg := startColReg
            rowPtrReg := rowPtrReg + 1.U

            ldRowIdxReg := 0.U
            ldBeatIdxReg := 0.U
            ldOutstandingReg := false.B
            state := load_next_row
          }.otherwise {
            colPtrReg := colPtrReg + 1.U
            genElemIdxReg := 0.U
            state := generate_window
          }
        }
      }
    }

    is(load_next_row) {
      val canIssue = !ldOutstandingReg && (ldBeatIdxReg < inColWords)
      val rowElem = rowPtrReg + kRowReg - 1.U
      val reqAddr = rBaseBeatReg + rowElem * inColWords + ldBeatIdxReg
      val targetSlot = rowFifo.io.slotToOverwrite

      io.bankRead(0).io.req.valid     := canIssue
      io.bankRead(0).io.req.bits.addr := reqAddr
      io.bankRead(0).io.resp.ready    := ldOutstandingReg

      when(io.bankRead(0).io.req.fire) {
        ldOutstandingReg := true.B
      }

      when(io.bankRead(0).io.resp.fire) {
        lineBuffer(targetSlot)(ldBeatIdxReg) := io.bankRead(0).io.resp.bits.data.asUInt
        ldOutstandingReg := false.B

        when(ldBeatIdxReg + 1.U === inColWords) {
          ldBeatIdxReg := 0.U
          rowFifo.io.advance := true.B
          genElemIdxReg := 0.U
          state := generate_window
        }.otherwise {
          ldBeatIdxReg := ldBeatIdxReg + 1.U
        }
      }
    }

    is(complete) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }
}
