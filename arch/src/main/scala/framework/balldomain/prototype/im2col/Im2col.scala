package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.im2col.configs.Im2colBallParam

/**
 * Im2col — FSM scheduler that coordinates LineBufferManager and StreamWriter.
 *
 * Optimizations applied:
 *   A) Eliminated elemBuffer — elements stream directly from lineBuffer to pack register
 *   B) Eliminated hardware divider — dual counters (kRowIdx, kColIdx) replace t/kCol, t%kCol
 */
@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  private val maxK = Im2colBallParam().InputNum

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

  // --- Sub-modules ---
  val lineBuf: Instance[LineBufferManager] = Instantiate(new LineBufferManager(b))
  val writer:  Instance[StreamWriter]      = Instantiate(new StreamWriter(b))

  // --- FSM ---
  val idle :: preload :: stream :: flushing :: loadNext :: complete :: Nil = Enum(6)
  val state                                                                = RegInit(idle)

  // --- Registers ---
  private val robIdReg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSubReg     = RegInit(false.B)
  private val subRobIdReg  = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBankReg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBankReg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val rBaseBeatReg = RegInit(0.U(32.W))
  private val wBaseBeatReg = RegInit(0.U(32.W))

  private val kRowReg     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kColReg     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val inRowReg    = RegInit(0.U(16.W))
  private val inColReg    = RegInit(0.U(16.W))
  private val startRowReg = RegInit(0.U(16.W))
  private val startColReg = RegInit(0.U(16.W))
  private val rowPtrReg   = RegInit(0.U(16.W))
  private val colPtrReg   = RegInit(0.U(16.W))

  // Dual counters (optimization B — replaces hardware divider)
  private val kRowIdxReg  = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kColIdxReg  = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val elemDoneReg = RegInit(false.B)

  // --- Derived signals ---
  private val rowMax       = inRowReg - kRowReg
  private val colMax       = inColReg - kColReg
  private val rowEnd       = rowPtrReg === (startRowReg + rowMax)
  private val colEnd       = colPtrReg === (startColReg + colMax)
  private val isLastWindow = rowEnd && colEnd

  // --- Top-level IO defaults ---
  io.cmdReq.ready            := (state === idle)
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg
  io.status.idle             := (state === idle)
  io.status.running          := (state =/= idle) && (state =/= complete)

  // --- Wire up LineBufferManager ---
  for (i <- 0 until inBW) {
    lineBuf.io.bankRead(i) <> io.bankRead(i)
  }
  lineBuf.io.startPreload    := false.B
  lineBuf.io.startLoadNext   := false.B
  lineBuf.io.kRow            := kRowReg
  lineBuf.io.inCol           := inColReg
  lineBuf.io.rowPtr          := rowPtrReg
  lineBuf.io.rBaseBeat       := rBaseBeatReg
  lineBuf.io.rBankId         := rBankReg
  lineBuf.io.robId           := robIdReg
  lineBuf.io.elemReq.kRowIdx := kRowIdxReg
  lineBuf.io.elemReq.kColIdx := kColIdxReg
  lineBuf.io.elemReq.colPtr  := colPtrReg

  // --- Wire up StreamWriter ---
  for (i <- 0 until outBW) {
    writer.io.bankWrite(i) <> io.bankWrite(i)
  }
  writer.io.start     := false.B
  writer.io.init      := false.B
  writer.io.flush     := false.B
  writer.io.wBaseBeat := wBaseBeatReg
  writer.io.wBankId   := wBankReg
  writer.io.robId     := robIdReg

  // Element stream: connect lineBuffer output to writer input
  writer.io.elemIn.valid := false.B
  writer.io.elemIn.bits  := lineBuf.io.elemData

  // --- FSM ---
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        robIdReg    := io.cmdReq.bits.rob_id
        isSubReg    := io.cmdReq.bits.is_sub
        subRobIdReg := io.cmdReq.bits.sub_rob_id
        rBankReg    := io.cmdReq.bits.cmd.op1_bank
        wBankReg    := io.cmdReq.bits.cmd.wr_bank

        kColReg     := io.cmdReq.bits.cmd.special(3, 0)
        kRowReg     := io.cmdReq.bits.cmd.special(7, 4)
        inColReg    := io.cmdReq.bits.cmd.special(12, 8)
        inRowReg    := io.cmdReq.bits.cmd.special(22, 13)
        startColReg := io.cmdReq.bits.cmd.special(27, 23)
        startRowReg := io.cmdReq.bits.cmd.special(37, 28)

        rowPtrReg := io.cmdReq.bits.cmd.special(37, 28)
        colPtrReg := io.cmdReq.bits.cmd.special(27, 23)

        rBaseBeatReg := 0.U
        wBaseBeatReg := 0.U
        kRowIdxReg   := 0.U
        kColIdxReg   := 0.U
        elemDoneReg  := false.B

        val cmdKCol      = io.cmdReq.bits.cmd.special(3, 0)
        val cmdKRow      = io.cmdReq.bits.cmd.special(7, 4)
        val cmdInCol     = io.cmdReq.bits.cmd.special(12, 8)
        val cmdInRow     = io.cmdReq.bits.cmd.special(22, 13)
        val invalidShape = (cmdKCol === 0.U) || (cmdKRow === 0.U) || (cmdInCol === 0.U) || (cmdInRow === 0.U) ||
          (cmdInCol < cmdKCol) || (cmdInRow < cmdKRow)

        when(invalidShape) {
          state := complete
        }.otherwise {
          lineBuf.io.startPreload := true.B
          state                   := preload
        }
      }
    }

    is(preload) {
      when(lineBuf.io.loadDone) {
        kRowIdxReg     := 0.U
        kColIdxReg     := 0.U
        elemDoneReg    := false.B
        writer.io.init := true.B
        state          := stream
      }
    }

    is(stream) {
      // Stream elements from lineBuffer through writer
      when(!elemDoneReg && writer.io.elemIn.ready) {
        writer.io.elemIn.valid := true.B

        // Advance dual counters
        val isLastElem = (kRowIdxReg === (kRowReg - 1.U)) && (kColIdxReg === (kColReg - 1.U))
        when(isLastElem) {
          elemDoneReg := true.B
        }.otherwise {
          when(kColIdxReg === (kColReg - 1.U)) {
            kColIdxReg := 0.U
            kRowIdxReg := kRowIdxReg + 1.U
          }.otherwise {
            kColIdxReg := kColIdxReg + 1.U
          }
        }
      }

      // Window done — transition without flushing (pack continues across windows)
      when(elemDoneReg && !writer.io.busy) {
        when(isLastWindow) {
          // Last window: flush remaining partial pack
          writer.io.flush := true.B
          state           := flushing
        }.elsewhen(colEnd) {
          // Row boundary: need to load next row
          colPtrReg                := startColReg
          rowPtrReg                := rowPtrReg + 1.U
          kRowIdxReg               := 0.U
          kColIdxReg               := 0.U
          elemDoneReg              := false.B
          lineBuf.io.startLoadNext := true.B
          state                    := loadNext
        }.otherwise {
          // Column slide: directly start next window
          colPtrReg   := colPtrReg + 1.U
          kRowIdxReg  := 0.U
          kColIdxReg  := 0.U
          elemDoneReg := false.B
          state       := stream
        }
      }
    }

    is(flushing) {
      // Wait for writer to finish flushing (write request must fire)
      when(!writer.io.busy) {
        state := complete
      }
    }

    is(loadNext) {
      when(lineBuf.io.loadDone) {
        kRowIdxReg  := 0.U
        kColIdxReg  := 0.U
        elemDoneReg := false.B
        state       := stream
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
