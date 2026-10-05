package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class Mxfp8Decode extends Module {

  @public
  val io = IO(new Bundle {
    val code    = Input(UInt(8.W))
    val scale   = Input(UInt(8.W))
    val out     = Output(UInt(32.W))
    val invalid = Output(Bool())
  })

  val exponent       = io.code(6, 3)
  val mantissa       = Mux(exponent === 0.U, Cat(0.U(1.W), io.code(2, 0)), Cat(1.U(1.W), io.code(2, 0)))
  val leading        = 3.U - PriorityEncoder(Reverse(mantissa))
  val binaryExponent = Mux(exponent === 0.U, 1.U, exponent) +& io.scale
  val ieeeExponent   = (binaryExponent +& leading).zext - 10.S
  val normalFraction = (mantissa.pad(32) << (23.U - leading))(22, 0)
  val subnormalShift = binaryExponent +& 12.U
  val subnormal      = (mantissa.pad(32) << subnormalShift(4, 0))(30, 0)
  val normal         = Cat(ieeeExponent.asUInt(7, 0), normalFraction)
  io.out     := Cat(io.code(7), Mux(mantissa === 0.U, 0.U(31.W), Mux(ieeeExponent > 0.S, normal, subnormal)))
  io.invalid := io.code(6, 0) === 127.U || io.scale === 255.U || (mantissa =/= 0.U && ieeeExponent >= 255.S)
}
