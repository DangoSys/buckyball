package examples.balls.mxquant

import chisel3._
import chisel3.util._

class Encode extends Module {

  val io = IO(new Bundle {
    val input         = Input(UInt(32.W))
    val blockExponent = Input(SInt(10.W))
    val code          = Output(UInt(8.W))
  })

  val fraction           = io.input(22, 0)
  val normalization      = PriorityEncoder(Reverse(fraction)) +& 1.U
  val normalizedFraction = (fraction << normalization(4, 0))(22, 0)
  val rawExponent        = io.input(30, 23)
  val exponent           = Mux(rawExponent === 0.U, 1.S(10.W) - normalization.zext, rawExponent.zext)
  val mantissa           = Mux(rawExponent === 0.U, normalizedFraction, fraction)
  val effective          = exponent - io.blockExponent - 120.S(10.W)

  val subnormalShift = (21.S(10.W) - effective).asUInt
  val shift          = subnormalShift(4, 0)
  val significand    = Cat(1.U(1.W), mantissa).pad(32)
  val half           = (1.U(32.W) << (shift - 1.U))(31, 0)
  val rounded        = (significand + half - 1.U + ((significand >> shift) & 1.U)) >> shift
  val subnormalCode  = Mux(subnormalShift > 31.U, 0.U, rounded)
  val normalRounded  = (mantissa.pad(32) + 524287.U + mantissa(20)) >> 20
  val normalCode     = (effective.asUInt << 3) + normalRounded
  val code           = Mux(effective <= 0.S, subnormalCode, normalCode)
  val clipped        = Mux(code > 126.U, 126.U(7.W), code(6, 0))
  io.code := Cat(io.input(31), Mux(io.input(30, 0) === 0.U, 0.U(7.W), clipped))
}
