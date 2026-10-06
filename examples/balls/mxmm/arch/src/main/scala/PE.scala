package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import freechips.rocketchip.tile.MulAddRecFNPipe
import hardfloat._

@instantiable
class PE(latency: Int) extends Module {
  require(latency == 3)

  @public
  val io = IO(new Bundle {
    val valid    = Input(Bool())
    val separate = Input(Bool())
    val a        = Input(UInt(32.W))
    val b        = Input(UInt(32.W))
    val c        = Input(UInt(33.W))
    val out      = Output(UInt(33.W))
    val outValid = Output(Bool())
  })

  val multiply = Module(new MulAddRecFNPipe(latency, 8, 24))
  multiply.io.validin        := io.valid
  multiply.io.op             := 0.U
  multiply.io.a              := recFNFromFN(8, 24, io.a)
  multiply.io.b              := recFNFromFN(8, 24, io.b)
  multiply.io.c              := Mux(io.separate, Cat(io.a(31) ^ io.b(31), 0.U(32.W)), io.c)
  multiply.io.roundingMode   := consts.round_near_even
  multiply.io.detectTininess := consts.tininess_afterRounding.asUInt

  val productMode = Pipe(io.valid, io.separate, latency)
  val previous    = Pipe(io.valid && io.separate, io.c, latency)
  val add         = Instantiate(new AddPipe)
  add.io.valid := multiply.io.validout && productMode.bits
  add.io.a     := multiply.io.out
  add.io.b     := previous.bits
  when(add.io.valid) {
    assert(fNFromRecFN(8, 24, multiply.io.out)(30, 23) =/= 255.U, "MXMM_F32 product must remain finite")
  }
  val fusedValid = multiply.io.validout && !productMode.bits
  io.outValid := fusedValid || add.io.outValid
  io.out      := Mux(add.io.outValid, add.io.out, multiply.io.out)
  when(io.outValid) {
    assert(fNFromRecFN(8, 24, io.out)(30, 23) =/= 255.U, "MATMUL accumulation must remain finite")
  }
}
