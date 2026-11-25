package examples.designs.piplus.balls.softmax

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

/** One Horner step of P(f): p * f, then + c, each rounded FP32 and registered. f, n and `zero` ride along. */
@instantiable
class Horner(c: Long) extends Module {

  @public
  val io = IO(new Bundle {
    val advance = Input(Bool())
    val inValid = Input(Bool())
    val inMask  = Input(Bool())
    val inP     = Input(UInt(33.W))
    val inF     = Input(UInt(33.W))
    val inN     = Input(UInt(8.W))
    val inZero  = Input(Bool())

    val outValid = Output(Bool())
    val outMask  = Output(Bool())
    val outP     = Output(UInt(33.W))
    val outF     = Output(UInt(33.W))
    val outN     = Output(UInt(8.W))
    val outZero  = Output(Bool())
  })

  val mul = Module(new MulRecFN(8, 24))
  mul.io.a              := io.inP
  mul.io.b              := io.inF
  mul.io.roundingMode   := consts.round_near_even
  mul.io.detectTininess := consts.tininess_afterRounding

  val valid1   = RegEnable(io.inValid, false.B, io.advance)
  val mask1    = RegEnable(io.inMask, io.advance)
  val product1 = RegEnable(mul.io.out, io.advance)
  val f1       = RegEnable(io.inF, io.advance)
  val n1       = RegEnable(io.inN, io.advance)
  val zero1    = RegEnable(io.inZero, io.advance)

  val add = Module(new AddRecFN(8, 24))
  add.io.subOp          := false.B
  add.io.a              := product1
  add.io.b              := recFNFromFN(8, 24, c.U(32.W))
  add.io.roundingMode   := consts.round_near_even
  add.io.detectTininess := consts.tininess_afterRounding

  io.outValid := RegEnable(valid1, false.B, io.advance)
  io.outMask  := RegEnable(mask1, io.advance)
  io.outP     := RegEnable(add.io.out, io.advance)
  io.outF     := RegEnable(f1, io.advance)
  io.outN     := RegEnable(n1, io.advance)
  io.outZero  := RegEnable(zero1, io.advance)
}
