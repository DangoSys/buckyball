package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._

class Im2colWindow(maxK: Int) extends Module {

  val io = IO(new Bundle {
    val init       = Input(Bool())
    val nextCol    = Input(Bool())
    val nextRow    = Input(Bool())
    val elemFire   = Input(Bool())
    val kRow       = Input(UInt(log2Ceil(maxK + 1).W))
    val kCol       = Input(UInt(log2Ceil(maxK + 1).W))
    val inRow      = Input(UInt(16.W))
    val inCol      = Input(UInt(16.W))
    val startRow   = Input(UInt(16.W))
    val startCol   = Input(UInt(16.W))
    val rowPtr     = Output(UInt(16.W))
    val colPtr     = Output(UInt(16.W))
    val kRowIdx    = Output(UInt(log2Ceil(maxK + 1).W))
    val kColIdx    = Output(UInt(log2Ceil(maxK + 1).W))
    val elemLast   = Output(Bool())
    val colEnd     = Output(Bool())
    val lastWindow = Output(Bool())
  })

  private val rowPtr  = RegInit(0.U(16.W))
  private val colPtr  = RegInit(0.U(16.W))
  private val kRowIdx = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kColIdx = RegInit(0.U(log2Ceil(maxK + 1).W))

  val rowMax   = io.inRow - io.kRow
  val colMax   = io.inCol - io.kCol
  val rowEnd   = rowPtr === (io.startRow + rowMax)
  val colEnd   = colPtr === (io.startCol + colMax)
  val elemLast = (kRowIdx === (io.kRow - 1.U)) && (kColIdx === (io.kCol - 1.U))

  when(io.init) {
    rowPtr  := io.startRow
    colPtr  := io.startCol
    kRowIdx := 0.U
    kColIdx := 0.U
  }.elsewhen(io.nextRow) {
    rowPtr  := rowPtr + 1.U
    colPtr  := io.startCol
    kRowIdx := 0.U
    kColIdx := 0.U
  }.elsewhen(io.nextCol) {
    colPtr  := colPtr + 1.U
    kRowIdx := 0.U
    kColIdx := 0.U
  }.elsewhen(io.elemFire && !elemLast) {
    when(kColIdx === (io.kCol - 1.U)) {
      kColIdx := 0.U
      kRowIdx := kRowIdx + 1.U
    }.otherwise {
      kColIdx := kColIdx + 1.U
    }
  }

  io.rowPtr     := rowPtr
  io.colPtr     := colPtr
  io.kRowIdx    := kRowIdx
  io.kColIdx    := kColIdx
  io.elemLast   := elemLast
  io.colEnd     := colEnd
  io.lastWindow := rowEnd && colEnd
}
