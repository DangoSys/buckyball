package examples.balls.f32add

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat.{fNFromRecFN, recFNFromFN, AddRecFN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}

@instantiable
class F32AddRow extends Module {

  @public val io = IO(new Bundle {
    val source      = Input(UInt(128.W))
    val accumulator = Input(UInt(128.W))
    val sum         = Output(UInt(128.W))
  })

  val result = Wire(Vec(4, UInt(32.W)))
  for (lane <- 0 until 4) {
    val add = Module(new AddRecFN(8, 24))
    add.io.subOp          := false.B
    add.io.a              := recFNFromFN(8, 24, io.accumulator(32 * lane + 31, 32 * lane))
    add.io.b              := recFNFromFN(8, 24, io.source(32 * lane + 31, 32 * lane))
    add.io.roundingMode   := round_near_even
    add.io.detectTininess := tininess_afterRounding
    val bits = fNFromRecFN(8, 24, add.io.out)
    val nan  = bits(30, 23).andR && bits(22, 0).orR
    result(lane) := Mux(nan, "h7fc00000".U(32.W), bits)
  }
  io.sum := Cat(result.reverse)
}
