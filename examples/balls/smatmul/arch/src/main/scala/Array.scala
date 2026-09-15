package examples.balls.smatmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import examples.balls.smatmul.configs.SMatMulBallParam
import framework.top.GlobalConfig

@instantiable
class Array(val b: GlobalConfig) extends Module {
  private val param = SMatMulBallParam(b)
  private val rows  = param.tileRows
  private val cols  = param.tileCols

  @public
  val io = IO(new Bundle {
    val start  = Input(Bool())
    val aRows  = Input(Vec(rows, UInt(128.W)))
    val bRows  = Input(Vec(cols, UInt(128.W)))
    val done   = Output(Bool())
    val result = Output(Vec(rows, UInt(512.W)))
  })

  val running = RegInit(false.B)
  val cycle   = RegInit(0.U(6.W))
  val pe      = Seq.tabulate(rows, cols)((_, _) => Instantiate(new PE))
  val aShift  = Reg(Vec(rows, UInt(128.W)))
  // The PE product register adds one cycle between the systolic operands and
  // the accumulator.  Keep the final drain cycle explicit so the last tile
  // product is included before done is asserted.
  val lastCyc = (rows + cols + 15).U

  when(io.start) {
    assert(!running, "Array: start while array is running")
    running := true.B
    cycle   := 0.U
    aShift  := io.aRows
  }.elsewhen(running) {
    when(cycle === lastCyc) {
      running := false.B
    }.otherwise {
      cycle := cycle + 1.U
    }
    for (row <- 0 until rows) {
      when(cycle >= row.U && cycle < (row + cols).U) {
        aShift(row) := aShift(row) >> 8
      }
    }
  }

  for (row <- 0 until rows) {
    val aWord  = aShift(row)(7, 0).asSInt
    val aValid = running && cycle >= row.U && cycle < (row + cols).U

    for (column <- 0 until cols) {
      pe(row)(column).io.clear := io.start
      if (column == 0) {
        pe(row)(column).io.aIn      := aWord
        pe(row)(column).io.aInValid := aValid
      } else {
        pe(row)(column).io.aIn      := pe(row)(column - 1).io.aOut
        pe(row)(column).io.aInValid := pe(row)(column - 1).io.aOutValid
      }

      val bCycle = cycle - column.U
      val bRow   = bCycle(3, 0)
      val bWord  = io.bRows(bRow)(8 * column + 7, 8 * column).asSInt
      val bValid = running && cycle >= column.U && cycle < (column + cols).U
      if (row == 0) {
        pe(row)(column).io.bIn      := bWord
        pe(row)(column).io.bInValid := bValid
      } else {
        pe(row)(column).io.bIn      := pe(row - 1)(column).io.bOut
        pe(row)(column).io.bInValid := pe(row - 1)(column).io.bOutValid
      }
    }
  }

  for (row <- 0 until rows) {
    io.result(row) := Cat((0 until cols).reverse.map(column => pe(row)(column).io.sum.asUInt))
  }
  io.done := running && cycle === lastCyc
}
