package examples.balls.im2col

import chisel3._
import chisel3.util._

class Im2colWindow(maxK: Int, iterW: Int) extends Module {
  require(iterW >= 1, "Im2colWindow iterW must be >= 1")

  val io = IO(new Bundle {
    val init      = Input(Bool())
    val next      = Input(Bool())
    val elemFire  = Input(Bool())
    val inRows    = Input(UInt(16.W))
    val inCols    = Input(UInt(16.W))
    val kRows     = Input(UInt(log2Ceil(maxK + 1).W))
    val kCols     = Input(UInt(log2Ceil(maxK + 1).W))
    val rowStride = Input(UInt(8.W))
    val colStride = Input(UInt(8.W))
    val padding   = Input(UInt(8.W))
    val startRow  = Input(UInt(8.W))
    val startCol  = Input(UInt(8.W))
    val outRow    = Output(UInt(16.W))
    val outCol    = Output(UInt(16.W))
    val kRowIdx   = Output(UInt(log2Ceil(maxK + 1).W))
    val kColIdx   = Output(UInt(log2Ceil(maxK + 1).W))
    val elemLast  = Output(Bool())
    val last      = Output(Bool())
  })

  private val outRowReg  = RegInit(0.U(iterW.W))
  private val outColReg  = RegInit(0.U(iterW.W))
  private val kRowIdxReg = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kColIdxReg = RegInit(0.U(log2Ceil(maxK + 1).W))

  private val paddedRows = io.inRows +& (io.padding << 1)
  private val paddedCols = io.inCols +& (io.padding << 1)
  private val outputRows =
    ((paddedRows - io.kRows - io.startRow) / io.rowStride) + 1.U
  private val outputCols =
    ((paddedCols - io.kCols - io.startCol) / io.colStride) + 1.U
  private val elemLast   =
    (kRowIdxReg === io.kRows - 1.U) && (kColIdxReg === io.kCols - 1.U)

  when(io.init) {
    outRowReg  := 0.U
    outColReg  := 0.U
    kRowIdxReg := 0.U
    kColIdxReg := 0.U
  }.elsewhen(io.next) {
    kRowIdxReg := 0.U
    kColIdxReg := 0.U
    when(outColReg + 1.U === outputCols) {
      outRowReg := outRowReg + 1.U
      outColReg := 0.U
    }.otherwise {
      outColReg := outColReg + 1.U
    }
  }.elsewhen(io.elemFire && !elemLast) {
    when(kColIdxReg === io.kCols - 1.U) {
      kColIdxReg := 0.U
      kRowIdxReg := kRowIdxReg + 1.U
    }.otherwise {
      kColIdxReg := kColIdxReg + 1.U
    }
  }

  io.outRow   := outRowReg
  io.outCol   := outColReg
  io.kRowIdx  := kRowIdxReg
  io.kColIdx  := kColIdxReg
  io.elemLast := elemLast
  io.last     := (outRowReg === outputRows - 1.U) && (outColReg === outputCols - 1.U)
}
