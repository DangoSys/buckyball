package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.BankRead
import examples.balls.im2col.configs.Im2colBallParam
import framework.top.GlobalConfig

@instantiable
class LineBufferManager(val b: GlobalConfig) extends Module {
  private val ballCfg      = Im2colBallParam(b)
  private val maxIter      = ballCfg.maxIter
  private val maxKSize     = ballCfg.maxKSize
  private val elemWidth    = ballCfg.inputWidth
  private val bankWidth    = b.memDomain.bankWidth
  private val lanesPerBeat = bankWidth / elemWidth
  require(isPow2(lanesPerBeat), "LineBuffer lanesPerBeat must be power of 2")
  private val maxBeats     = maxIter * maxIter
  require(isPow2(maxBeats) && maxBeats >= 2, "LineBuffer depth must be an even power of two")
  private val halfBeats    = maxBeats / 2
  private val kW           = log2Ceil(maxKSize + 1)
  private val addrW        = log2Ceil(b.memDomain.bankEntries)

  private val map = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(
      throw new IllegalArgumentException("Im2colBall not found in config")
    )

  private val inBW = map.inBW

  @public val io = IO(new Bundle {
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val start     = Input(Bool())
    val inRows    = Input(UInt(16.W))
    val inCols    = Input(UInt(16.W))
    val rowStride = Input(UInt(8.W))
    val colStride = Input(UInt(8.W))
    val padding   = Input(UInt(8.W))
    val startRow  = Input(UInt(8.W))
    val startCol  = Input(UInt(8.W))
    val inputBase = Input(UInt(6.W))
    val lane      = Input(UInt(log2Ceil(lanesPerBeat).W))
    val outRow    = Input(UInt(16.W))
    val outCol    = Input(UInt(16.W))
    val kRowIdx   = Input(UInt(kW.W))
    val kColIdx   = Input(UInt(kW.W))
    val rBankId   = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId     = Input(UInt(log2Up(b.frontend.rob_entries).W))
    val loadDone  = Output(Bool())
    val elemData  = Output(UInt(elemWidth.W))
  })

  private val bufLo    = SyncReadMem(halfBeats, UInt(bankWidth.W))
  private val bufHi    = SyncReadMem(halfBeats, UInt(bankWidth.W))
  private val active   = RegInit(false.B)
  private val pending  = RegInit(false.B)
  private val beat     = RegInit(0.U(addrW.W))
  private val dimW     = log2Ceil(maxIter + 1)
  private val beatsReg = RegInit(0.U((addrW + 1).W))

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := io.rBankId
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }

  io.bankRead(0).io.req.valid     := active && !pending
  io.bankRead(0).io.req.bits.addr := (io.inputBase +& beat).asTypeOf(UInt(addrW.W))
  io.bankRead(0).io.resp.ready    := pending
  io.loadDone                     := !active

  when(io.start) {
    val rows = io.inRows(dimW - 1, 0)
    val cols = io.inCols(dimW - 1, 0)
    beatsReg := rows * cols
    active   := true.B
    pending  := false.B
    beat     := 0.U
  }.elsewhen(io.bankRead(0).io.req.fire) {
    pending := true.B
  }

  when(io.bankRead(0).io.resp.fire) {
    pending := false.B
    when(beat +& 1.U === beatsReg) {
      active := false.B
    }.otherwise {
      beat := beat + 1.U
    }
  }

  private val paddedRow   = io.startRow +& (io.outRow(dimW - 1, 0) * io.rowStride) +& io.kRowIdx
  private val paddedCol   = io.startCol +& (io.outCol(dimW - 1, 0) * io.colStride) +& io.kColIdx
  private val rowValid    =
    paddedRow >= io.padding && paddedRow < io.padding + io.inRows
  private val colValid    =
    paddedCol >= io.padding && paddedCol < io.padding + io.inCols
  private val inBound     = rowValid && colValid
  private val srcRow      = Mux(inBound, paddedRow - io.padding, 0.U)
  private val srcCol      = Mux(inBound, paddedCol - io.padding, 0.U)
  private val elemIndex   = srcRow(dimW - 1, 0) * io.inCols(dimW - 1, 0) + srcCol(dimW - 1, 0)
  private val beatIndex   = elemIndex
  private val laneIndex   = io.lane
  private val memoryWrite = io.bankRead(0).io.resp.fire
  private val memoryRead  = !active

  private val memoryAddress = Mux(
    memoryWrite,
    beat(log2Ceil(maxBeats) - 1, 0),
    beatIndex(log2Ceil(maxBeats) - 1, 0)
  )

  private val memoryBank   = memoryAddress(log2Ceil(maxBeats) - 1)
  private val memoryRow    = memoryAddress(log2Ceil(halfBeats) - 1, 0)
  private val memoryEnable = memoryWrite || memoryRead

  private val loData = bufLo.readWrite(
    memoryRow,
    io.bankRead(0).io.resp.bits.data.asUInt,
    memoryEnable && !memoryBank,
    memoryWrite
  )

  private val hiData = bufHi.readWrite(
    memoryRow,
    io.bankRead(0).io.resp.bits.data.asUInt,
    memoryEnable && memoryBank,
    memoryWrite
  )

  private val readBank   = RegEnable(memoryBank, false.B, memoryRead)
  private val memoryData = Mux(readBank, hiData, loData)
  private val laneR      = RegInit(0.U(log2Ceil(lanesPerBeat).W))
  private val boundR     = RegInit(false.B)
  private val beatData   = RegInit(0.U(bankWidth.W))
  private val laneD      = RegInit(0.U(log2Ceil(lanesPerBeat).W))
  private val boundD     = RegInit(false.B)
  private val elemHold   = RegInit(0.U(elemWidth.W))
  laneR    := laneIndex
  boundR   := inBound
  beatData := memoryData
  laneD    := laneR
  boundD   := boundR
  private val lanes =
    beatData.asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))
  elemHold := Mux(boundD, lanes(laneD), 0.U)

  io.elemData := elemHold
}
