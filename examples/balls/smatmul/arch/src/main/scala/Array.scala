package examples.balls.smatmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

@instantiable
class Array extends Module {
  private val tile = 16

  @public
  val io = IO(new Bundle {
    val start  = Input(Bool())
    val aRows  = Input(Vec(tile, UInt(128.W)))
    val bRows  = Input(Vec(tile, UInt(128.W)))
    val done   = Output(Bool())
    val result = Output(Vec(tile, UInt(512.W)))
  })

  val running = RegInit(false.B)
  val cycle   = RegInit(0.U(6.W))
  val pe      = Seq.tabulate(tile, tile)((_, _) => Instantiate(new PE))

  when(io.start) {
    assert(!running, "Array: start while array is running")
    running := true.B
    cycle   := 0.U
  }.elsewhen(running) {
    when(cycle === 45.U) {
      running := false.B
    }.otherwise {
      cycle := cycle + 1.U
    }
  }

  for (row <- 0 until tile) {
    val aCycle = cycle - row.U
    val aWord  = (io.aRows(row) >> (aCycle << 3))(7, 0).asSInt
    val aValid = running && cycle >= row.U && cycle < (row + tile).U

    for (column <- 0 until tile) {
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
      val bWord  = (io.bRows(bRow)(8 * column + 7, 8 * column)).asSInt
      val bValid = running && cycle >= column.U && cycle < (column + tile).U
      if (row == 0) {
        pe(row)(column).io.bIn      := bWord
        pe(row)(column).io.bInValid := bValid
      } else {
        pe(row)(column).io.bIn      := pe(row - 1)(column).io.bOut
        pe(row)(column).io.bInValid := pe(row - 1)(column).io.bOutValid
      }
    }
  }

  for (row <- 0 until tile) {
    io.result(row) := Cat((0 until tile).reverse.map(column => pe(row)(column).io.sum.asUInt))
  }
  io.done := running && cycle === 45.U
}
