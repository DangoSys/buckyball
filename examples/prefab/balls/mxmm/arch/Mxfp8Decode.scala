package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

// An MXFP8 (E4M3) element with its block scale applied, exactly:
// (-1)^sign * sig * 2^exp, sig being the 4-bit significand with its leading
// bit. exp = max(e, 1) + scale - 137 lies in [-136, 132].
class Mxfp8Value extends Bundle {
  val sign = Bool()
  val sig  = UInt(4.W)
  val exp  = SInt(9.W)
}

@instantiable
class Mxfp8Decode extends Module {

  @public
  val io = IO(new Bundle {
    val code    = Input(UInt(8.W))
    val scale   = Input(UInt(8.W))
    val out     = Output(new Mxfp8Value)
    val invalid = Output(Bool())
  })

  val exponent       = io.code(6, 3)
  val mantissa       = Mux(exponent === 0.U, Cat(0.U(1.W), io.code(2, 0)), Cat(1.U(1.W), io.code(2, 0)))
  val binaryExponent = Mux(exponent === 0.U, 1.U, exponent) +& io.scale
  io.out.sign := io.code(7)
  io.out.sig  := mantissa
  io.out.exp  := binaryExponent.zext - 137.S
  // The element must be finite and fit FP32: its FP32 exponent field below 255.
  val leading      = 3.U - PriorityEncoder(Reverse(mantissa))
  val ieeeExponent = (binaryExponent +& leading).zext - 10.S
  io.invalid := io.code(6, 0) === 127.U || io.scale === 255.U || (mantissa =/= 0.U && ieeeExponent >= 255.S)
}
