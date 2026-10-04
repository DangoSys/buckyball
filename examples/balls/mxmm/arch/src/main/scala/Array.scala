package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import examples.balls.mxmm.configs.MxmmBallParam
import hardfloat._

@instantiable
class Array(p: MxmmBallParam) extends Module {

  @public
  val io = IO(new Bundle {
    val valid     = Input(Bool())
    val context   = Input(UInt(log2Ceil(p.contexts).W))
    val separate  = Input(Bool())
    val vector    = Input(Bool())
    val a         = Input(Vec(p.tileRows, UInt(32.W)))
    val b         = Input(Vec(p.tileCols, UInt(32.W)))
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
  val fusedTag     = Pipe(io.valid && !io.separate, io.context, p.arithmeticLatency)
  val separateTag  = Pipe(io.valid && io.separate, io.context, 2 * p.arithmeticLatency)
  io.completed := fusedTag.valid || separateTag.valid
  for {
    row <- 0 until p.tileRows
    col <- 0 until p.tileCols
  } {
    val pe = Instantiate(new PE(p.arithmeticLatency))
    pe.io.valid    := io.valid && (if (row == 0) true.B else !io.vector)
    pe.io.separate := io.separate
    pe.io.a        := io.a(row)
    pe.io.b        := io.b(col)
    pe.io.c        := accumulators(io.context)(row)(col)
    when(pe.io.outValid) {
      accumulators(Mux(separateTag.valid, separateTag.bits, fusedTag.bits))(row)(col) := pe.io.out
    }
  }
}
