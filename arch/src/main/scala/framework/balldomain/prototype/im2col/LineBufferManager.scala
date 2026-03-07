package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.BankRead
import framework.top.GlobalConfig
import framework.balldomain.prototype.im2col.configs.Im2colBallParam

/**
 * LineBufferManager — manages lineBuffer loading and element extraction.
 *
 * Handles preload (loading kRow rows) and load_next_row (loading 1 new row
 * with FIFO rotation). Provides a combinational element read port that
 * extracts a single element from the lineBuffer given (kRowIdx, kColIdx).
 */
@instantiable
class LineBufferManager(val b: GlobalConfig) extends Module {
  private val maxK          = Im2colBallParam().InputNum
  private val elemWidth     = Im2colBallParam().inputWidth
  private val bankWidth     = b.memDomain.bankWidth
  private val lanesPerBeat  = 16
  private val maxInCol      = 32
  private val maxInColWords = (maxInCol + lanesPerBeat - 1) / lanesPerBeat

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val inBW = mapping.inBW

  @public val io = IO(new Bundle {
    // SRAM read port (directly connected to Ball's bankRead)
    val bankRead = Vec(inBW, Flipped(new BankRead(b)))

    // Control inputs
    val startPreload  = Input(Bool()) // pulse: begin preloading kRow rows
    val startLoadNext = Input(Bool()) // pulse: begin loading 1 new row

    // Configuration (latched by Im2col on cmdReq.fire)
    val kRow      = Input(UInt(log2Ceil(maxK + 1).W))
    val inCol     = Input(UInt(16.W))
    val rowPtr    = Input(UInt(16.W))
    val rBaseBeat = Input(UInt(32.W))
    val rBankId   = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId     = Input(UInt(log2Up(b.frontend.rob_entries).W))

    // Status outputs
    val loadDone = Output(Bool()) // high when load operation is complete

    // Element read port (combinational)
    val elemReq = new Bundle {
      val kRowIdx = Input(UInt(log2Ceil(maxK + 1).W))
      val kColIdx = Input(UInt(log2Ceil(maxK + 1).W))
      val colPtr  = Input(UInt(16.W))
    }

    val elemData = Output(UInt(elemWidth.W))
  })

  private def ceilDiv(a: UInt, d: Int): UInt = (a + (d - 1).U) / d.U
  private val inColWords = ceilDiv(io.inCol, lanesPerBeat)

  // Line buffer storage
  private val lineBuffer = RegInit(VecInit(Seq.fill(maxK)(VecInit(Seq.fill(maxInColWords)(0.U(bankWidth.W))))))

  // Row slot FIFO for circular buffer management
  private val rowFifo = Module(new RowSlotFIFO(maxK))
  rowFifo.io.kRows   := io.kRow
  rowFifo.io.init    := false.B
  rowFifo.io.advance := false.B

  // Load state machine
  val sIdle :: sPreload :: sLoadNext :: Nil = Enum(3)
  val loadState                             = RegInit(sIdle)

  private val ldRowIdxReg      = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val ldBeatIdxReg     = RegInit(0.U(log2Ceil(maxInColWords + 1).W))
  private val ldOutstandingReg = RegInit(false.B)

  io.loadDone := (loadState === sIdle)

  // Default bankRead signals
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := io.rBankId
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }

  // Element extraction (combinational) — no division needed
  private val startLane    = io.elemReq.colPtr % lanesPerBeat.U
  private val physicalSlot = RowSlotFIFO.logicalToPhysical(rowFifo.io.head, io.elemReq.kRowIdx, io.kRow)
  private val laneSum      = startLane + io.elemReq.kColIdx
  private val beatIdx      = laneSum / lanesPerBeat.U
  private val laneIdx      = laneSum           % lanesPerBeat.U
  private val beatWord     = lineBuffer(physicalSlot)(beatIdx)
  private val lanes        = beatWord.asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))
  io.elemData := lanes(laneIdx)

  switch(loadState) {
    is(sIdle) {
      when(io.startPreload) {
        ldRowIdxReg      := 0.U
        ldBeatIdxReg     := 0.U
        ldOutstandingReg := false.B
        rowFifo.io.init  := true.B
        loadState        := sPreload
      }.elsewhen(io.startLoadNext) {
        ldBeatIdxReg     := 0.U
        ldOutstandingReg := false.B
        loadState        := sLoadNext
      }
    }

    is(sPreload) {
      val doneRows = ldRowIdxReg === io.kRow
      val canIssue = !doneRows && !ldOutstandingReg && (ldBeatIdxReg < inColWords)
      val rowElem  = io.rowPtr + ldRowIdxReg
      val reqAddr  = io.rBaseBeat + rowElem * inColWords + ldBeatIdxReg

      io.bankRead(0).io.req.valid     := canIssue
      io.bankRead(0).io.req.bits.addr := reqAddr
      io.bankRead(0).io.resp.ready    := ldOutstandingReg

      when(io.bankRead(0).io.req.fire) {
        ldOutstandingReg := true.B
      }

      when(io.bankRead(0).io.resp.fire) {
        lineBuffer(ldRowIdxReg)(ldBeatIdxReg) := io.bankRead(0).io.resp.bits.data.asUInt
        ldOutstandingReg                      := false.B

        when(ldBeatIdxReg + 1.U === inColWords) {
          ldBeatIdxReg := 0.U
          ldRowIdxReg  := ldRowIdxReg + 1.U
        }.otherwise {
          ldBeatIdxReg := ldBeatIdxReg + 1.U
        }
      }

      when(doneRows && !ldOutstandingReg) {
        loadState := sIdle
      }
    }

    is(sLoadNext) {
      val canIssue   = !ldOutstandingReg && (ldBeatIdxReg < inColWords)
      val rowElem    = io.rowPtr + io.kRow - 1.U
      val reqAddr    = io.rBaseBeat + rowElem * inColWords + ldBeatIdxReg
      val targetSlot = rowFifo.io.slotToOverwrite

      io.bankRead(0).io.req.valid     := canIssue
      io.bankRead(0).io.req.bits.addr := reqAddr
      io.bankRead(0).io.resp.ready    := ldOutstandingReg

      when(io.bankRead(0).io.req.fire) {
        ldOutstandingReg := true.B
      }

      when(io.bankRead(0).io.resp.fire) {
        lineBuffer(targetSlot)(ldBeatIdxReg) := io.bankRead(0).io.resp.bits.data.asUInt
        ldOutstandingReg                     := false.B

        when(ldBeatIdxReg + 1.U === inColWords) {
          ldBeatIdxReg       := 0.U
          rowFifo.io.advance := true.B
          loadState          := sIdle
        }.otherwise {
          ldBeatIdxReg := ldBeatIdxReg + 1.U
        }
      }
    }
  }
}
