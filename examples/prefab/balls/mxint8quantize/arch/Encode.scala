package examples.balls.mxint8quantize

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** One FP32 value to its INT8 code: clamp(round_half_even(x / 2^e), -127, 127). */
@instantiable
class Encode extends Module {

  @public
  val io = IO(new Bundle {
    val valid    = Input(Bool())
    val input    = Input(UInt(32.W))
    val exponent = Input(SInt(10.W))
    val code     = Output(UInt(8.W))
  })

  val field       = io.input(30, 23)
  val significand = Cat(field =/= 0.U, io.input(22, 0))
  val shift       = 150.S(11.W) + io.exponent - Mux(field === 0.U, 1.U, field).zext
  when(io.valid)(assert(shift >= 17.S, "MXINT8QUANT element exceeds its block maximum"))

  val extended  = Cat(significand, 0.U(32.W)) >> shift.asUInt(4, 0)
  val quotient  = extended(55, 32)
  val roundUp   = extended(31) && (extended(30, 0).orR || quotient(0))
  val rounded   = quotient +& roundUp
  val magnitude = Mux(shift > 31.S, 0.U, Mux(rounded > 127.U, 127.U, rounded))
  io.code := Mux(io.input(31), 0.U(8.W) - magnitude, magnitude)(7, 0)
}
