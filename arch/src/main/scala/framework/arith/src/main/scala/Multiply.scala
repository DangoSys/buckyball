package framework.arith

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

/** Eight radix-256 steps; one 64-by-8 partial product, not a combinational 64-by-64 multiplier. */
@instantiable
class Multiply extends Module {

  @public val io = IO(new Bundle {
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
    when(step === 7.U) {
      val product = Mux(negative, -next, next)
      result := Mux(high, product(127, 64), product(63, 0))
      state  := complete
    }
  }
  when(io.response.fire)(state := idle)
}
