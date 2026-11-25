package framework.generic.arith

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class MultiplyRequest extends Bundle {
  val a       = UInt(64.W)
  val b       = UInt(64.W)
  val signedA = Bool()
  val signedB = Bool()
  val high    = Bool()
}

/** Radix-256 partial products, with a declared magnitude width for operand B. */
@instantiable
class Multiply(multiplierBits: Int) extends Module {
  require(multiplierBits >= 8 && multiplierBits <= 64 && multiplierBits % 8 == 0)

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MultiplyRequest))
    val response = Decoupled(UInt(64.W))
  })

  val idle :: calculate :: complete :: Nil = Enum(3)
  val state                                = RegInit(idle)
  val a                                    = Reg(UInt(64.W))
  val b                                    = Reg(UInt(64.W))
  val sum                                  = Reg(UInt(128.W))
  val step                                 = Reg(UInt(3.W))
  val negative                             = Reg(Bool())
  val high                                 = Reg(Bool())
  val result                               = Reg(UInt(64.W))
  val negativeA                            = io.request.bits.signedA && io.request.bits.a(63)
  val negativeB                            = io.request.bits.signedB && io.request.bits.b(63)
  io.request.ready  := state === idle && !reset.asBool
  io.response.valid := state === complete && !reset.asBool
  io.response.bits  := result
  when(io.request.fire) {
    if (multiplierBits < 64) {
      assert((Mux(negativeB, -io.request.bits.b, io.request.bits.b) >> multiplierBits) === 0.U)
    }
    a        := Mux(negativeA, -io.request.bits.a, io.request.bits.a)
    b        := Mux(negativeB, -io.request.bits.b, io.request.bits.b)
    negative := negativeA ^ negativeB
    high     := io.request.bits.high
    sum      := 0.U
    step     := 0.U
    state    := calculate
  }
  val partial = a * b(7, 0)
  val next = sum + (partial.pad(128) << (step << 3))(127, 0)
  when(state === calculate) {
    sum  := next
    b    := b >> 8
    step := step + 1.U
    when(step === (multiplierBits / 8 - 1).U) {
      val product = Mux(negative, -next, next)
      result := Mux(high, product(127, 64), product(63, 0))
      state  := complete
    }
  }
  when(io.response.fire)(state := idle)
}
