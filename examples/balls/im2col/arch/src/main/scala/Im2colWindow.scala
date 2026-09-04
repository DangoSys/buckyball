package examples.balls.im2col

import chisel3._
import chisel3.util._

class Im2colWindow(maxK: Int) extends Module {
  private val kW = log2Ceil(maxK + 1)

  val io = IO(new Bundle {
    val init        = Input(Bool())
    val next        = Input(Bool())
    val elemFire    = Input(Bool())
    val kernel      = Input(UInt(kW.W))
    val outputCols  = Input(UInt(16.W))
    val startRow    = Input(UInt(16.W))
    val startCol    = Input(UInt(16.W))
    val windowCount = Input(UInt(12.W))
    val outRow      = Output(UInt(16.W))
    val outCol      = Output(UInt(16.W))
    val kRowIdx     = Output(UInt(kW.W))
    val kColIdx     = Output(UInt(kW.W))
    val elemLast    = Output(Bool())
    val last        = Output(Bool())
  })

  private val outRowReg = RegInit(0.U(16.W))
  private val outColReg = RegInit(0.U(16.W))
  private val kRowReg   = RegInit(0.U(kW.W))
  private val kColReg   = RegInit(0.U(kW.W))
  private val colsReg   = RegInit(0.U(16.W))
  private val remaining = RegInit(0.U(12.W))

  private val elemLast = kRowReg === io.kernel - 1.U && kColReg === io.kernel - 1.U

  when(io.init) {
    outRowReg := io.startRow
    outColReg := io.startCol
    kRowReg   := 0.U
    kColReg   := 0.U
    colsReg   := io.outputCols
    remaining := io.windowCount
  }.elsewhen(io.next) {
    assert(remaining > 1.U, "Im2colWindow cannot advance past the selected range")
    kRowReg   := 0.U
    kColReg   := 0.U
    remaining := remaining - 1.U
    when(outColReg + 1.U === colsReg) {
      outRowReg := outRowReg + 1.U
      outColReg := 0.U
    }.otherwise {
      outColReg := outColReg + 1.U
    }
  }.elsewhen(io.elemFire && !elemLast) {
    when(kColReg === io.kernel - 1.U) {
      kColReg := 0.U
      kRowReg := kRowReg + 1.U
    }.otherwise {
      kColReg := kColReg + 1.U
    }
  }

  io.outRow   := outRowReg
  io.outCol   := outColReg
  io.kRowIdx  := kRowReg
  io.kColIdx  := kColReg
  io.elemLast := elemLast
  io.last     := remaining === 1.U
}
