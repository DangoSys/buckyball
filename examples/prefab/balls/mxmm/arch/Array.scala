package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import examples.balls.mxmm.configs.MxmmBallParam
import hardfloat._

@instantiable
class Array(val ballParam: MxmmBallParam) extends Module {
  val p = ballParam

  @public
  val io = IO(new Bundle {
    val valid     = Input(Bool())
    val context   = Input(UInt(log2Ceil(p.contexts).W))
    val vector    = Input(Bool())
    val a         = Input(Vec(p.tileRows, new Mxfp8Value))
    val b         = Input(Vec(p.tileCols, new Mxfp8Value))
    val load      = Input(Bool())
    val row       = Input(UInt(4.W))
    val rowData   = Input(UInt(512.W))
    val rowOut    = Output(UInt(512.W))
    val completed = Output(Bool())
  })

  val accumulators = Reg(Vec(p.contexts, Vec(p.tileRows, Vec(p.tileCols, UInt(33.W)))))
  when(io.load) {
    for (col <- 0 until p.tileCols) {
      accumulators(io.context)(io.row)(col) := recFNFromFN(8, 24, io.rowData(32 * col + 31, 32 * col))
    }
  }
  io.rowOut := Cat((0 until p.tileCols).reverse.map(col => fNFromRecFN(8, 24, accumulators(io.context)(io.row)(col))))
  val tag          = Pipe(io.valid, io.context, p.arithmeticLatency)
  io.completed := tag.valid
  for {
    row <- 0 until p.tileRows
    col <- 0 until p.tileCols
  } {
    val pe =
      Instantiate(new PE(p.arithmeticLatency))
    pe.io.valid := io.valid && (if (row == 0) true.B else !io.vector)
    pe.io.a     := io.a(row)
    pe.io.b     := io.b(col)
    pe.io.c     := accumulators(io.context)(row)(col)
    when(pe.io.outValid) {
      accumulators(tag.bits)(row)(col) := pe.io.out
    }
  }
}
