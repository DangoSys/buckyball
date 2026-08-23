package examples.balls.im2col

import chisel3._
import chisel3.util._
import framework.balldomain.rs.BallRsIssue
import framework.top.GlobalConfig

class Im2colConfigRegs(
  val b:      GlobalConfig,
  maxIter:    Int,
  maxKSize:   Int,
  maxPadding: Int)
    extends Module {

  private val kW           = log2Ceil(maxKSize + 1)
  private val bankEntries  = b.memDomain.bankEntries
  private val bankNum      = b.memDomain.bankNum
  private val maxFootprint = bankEntries * bankNum
  private val tile         = 16

  val io = IO(new Bundle {
    val cmd       = Input(new BallRsIssue(b))
    val load      = Input(Bool())
    val invalid   = Output(Bool())
    val robId     = Output(UInt(log2Up(b.frontend.rob_entries).W))
    val isSub     = Output(Bool())
    val subRobId  = Output(UInt(log2Up(b.frontend.sub_rob_depth * 4).W))
    val rBank     = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wBank     = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val legacy    = Output(Bool())
    val inRows    = Output(UInt(16.W))
    val inCols    = Output(UInt(16.W))
    val kRows     = Output(UInt(kW.W))
    val kCols     = Output(UInt(kW.W))
    val rowStride = Output(UInt(8.W))
    val colStride = Output(UInt(8.W))
    val padding   = Output(UInt(8.W))
    val startRow  = Output(UInt(8.W))
    val startCol  = Output(UInt(8.W))
  })

  private val robId     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub     = RegInit(false.B)
  private val subRobId  = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBank     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBank     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val legacyReg = RegInit(false.B)
  private val inRows    = RegInit(0.U(16.W))
  private val inCols    = RegInit(0.U(16.W))
  private val kRows     = RegInit(0.U(kW.W))
  private val kCols     = RegInit(0.U(kW.W))
  private val rowStride = RegInit(1.U(8.W))
  private val colStride = RegInit(1.U(8.W))
  private val padding   = RegInit(0.U(8.W))
  private val startRow  = RegInit(0.U(8.W))
  private val startCol  = RegInit(0.U(8.W))

  val legacy       = io.cmd.cmd.iter === 0.U
  val cmdInRows    = Mux(legacy, io.cmd.cmd.special(31, 24), io.cmd.cmd.iter)
  val cmdInCols    = Mux(legacy, io.cmd.cmd.special(23, 16), io.cmd.cmd.iter)
  val cmdKRows     =
    Mux(legacy, io.cmd.cmd.special(15, 8), io.cmd.cmd.special(7, 0))
  val cmdKCols     = io.cmd.cmd.special(7, 0)
  val cmdRowStride = Mux(legacy, 1.U, io.cmd.cmd.special(15, 8))
  val cmdColStride =
    Mux(legacy, io.cmd.cmd.special(55, 48), io.cmd.cmd.special(15, 8))
  val cmdPadding   = Mux(legacy, 0.U, io.cmd.cmd.special(23, 16))
  val cmdStartRow  =
    Mux(legacy, io.cmd.cmd.special(47, 40), io.cmd.cmd.special(39, 32))
  val cmdStartCol  =
    Mux(legacy, io.cmd.cmd.special(39, 32), io.cmd.cmd.special(31, 24))
  val cmdRBank     = io.cmd.cmd.op1_bank
  val cmdWBank     = io.cmd.cmd.wr_bank

  when(io.load) {
    robId     := io.cmd.rob_id
    isSub     := io.cmd.is_sub
    subRobId  := io.cmd.sub_rob_id
    rBank     := cmdRBank
    wBank     := cmdWBank
    legacyReg := legacy
    inRows    := cmdInRows
    inCols    := cmdInCols
    kRows     := cmdKRows(kW - 1, 0)
    kCols     := cmdKCols(kW - 1, 0)
    rowStride := cmdRowStride
    colStride := cmdColStride
    padding   := cmdPadding
    startRow  := cmdStartRow
    startCol  := cmdStartCol
  }

  val paddedRows = cmdInRows +& (cmdPadding << 1)
  val paddedCols = cmdInCols +& (cmdPadding << 1)

  val shapeOk = (cmdInRows >= 1.U) && (cmdInRows <= maxIter.U) &&
    (cmdInCols >= 1.U) && (cmdInCols <= maxIter.U) &&
    (cmdKRows >= 1.U) && (cmdKRows <= maxKSize.U) &&
    (cmdKCols >= 1.U) && (cmdKCols <= maxKSize.U) &&
    (cmdRowStride >= 1.U) && (cmdColStride >= 1.U) &&
    (cmdPadding <= maxPadding.U) &&
    (cmdStartRow <= cmdPadding) && (cmdStartCol <= cmdPadding) &&
    (paddedRows >= cmdKRows + cmdStartRow) &&
    (paddedCols >= cmdKCols + cmdStartCol) &&
    (cmdRBank =/= cmdWBank)

  val outRows = Mux(
    shapeOk,
    ((paddedRows - cmdKRows - cmdStartRow) / cmdRowStride) + 1.U,
    0.U
  )

  val outCols = Mux(
    shapeOk,
    ((paddedCols - cmdKCols - cmdStartCol) / cmdColStride) + 1.U,
    0.U
  )

  val windows     = outRows * outCols
  val kElems      = cmdKRows * cmdKCols
  val mTiles      = (windows +& (tile - 1).U) / tile.U
  val kTiles      = (kElems +& (tile - 1).U) / tile.U
  val tiledRows   = mTiles * kTiles * tile.U
  val legacyBeats = (windows * kElems +& (tile - 1).U) / tile.U
  val footprint   = Mux(legacy, legacyBeats, tiledRows)

  // Matches emu capacity (groups * bank_lines); StreamWriter spans multi-group addrs.
  io.invalid   := !shapeOk || (footprint > maxFootprint.U)
  io.robId     := Mux(io.load, io.cmd.rob_id, robId)
  io.isSub     := Mux(io.load, io.cmd.is_sub, isSub)
  io.subRobId  := Mux(io.load, io.cmd.sub_rob_id, subRobId)
  io.rBank     := Mux(io.load, cmdRBank, rBank)
  io.wBank     := Mux(io.load, cmdWBank, wBank)
  io.legacy    := Mux(io.load, legacy, legacyReg)
  io.inRows    := Mux(io.load, cmdInRows, inRows)
  io.inCols    := Mux(io.load, cmdInCols, inCols)
  io.kRows     := Mux(io.load, cmdKRows(kW - 1, 0), kRows)
  io.kCols     := Mux(io.load, cmdKCols(kW - 1, 0), kCols)
  io.rowStride := Mux(io.load, cmdRowStride, rowStride)
  io.colStride := Mux(io.load, cmdColStride, colStride)
  io.padding   := Mux(io.load, cmdPadding, padding)
  io.startRow  := Mux(io.load, cmdStartRow, startRow)
  io.startCol  := Mux(io.load, cmdStartCol, startCol)
}
