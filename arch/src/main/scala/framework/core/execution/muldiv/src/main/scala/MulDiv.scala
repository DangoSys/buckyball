package framework.core.execution.muldiv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.execution.muldiv.configs.{Params => ModuleParam}

class Request(b: ModuleParam) extends Bundle {
  val p = b

  val a         = UInt(p.xLen.W)
  val rhs       = UInt(p.xLen.W)
  val operation = UInt(3.W)
  val word      = Bool()
}

/** RV64M multiplier and restoring divider; one operation owns the circuit until response. */
class MulDivIO(b: ModuleParam) extends Bundle {
  val p = b

  val request  = Flipped(Decoupled(new Request(b)))
  val response = Decoupled(UInt(p.xLen.W))

}

@instantiable
class MulDiv(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new MulDivIO(b))

  val idle :: divide :: respond :: Nil = Enum(3)
  val state                            = RegInit(idle)
  val quotient                         = Reg(UInt(64.W))
  val remainder                        = Reg(UInt(65.W))
  val divisor                          = Reg(UInt(65.W))
  val count                            = Reg(UInt(7.W))
  val negateQuotient                   = Reg(Bool())
  val negateRemainder                  = Reg(Bool())
  val returnRemainder                  = Reg(Bool())
  val word                             = Reg(Bool())
  val result                           = Reg(UInt(64.W))

  io.request.ready             := state === idle
  io.response.valid            := state === respond
  io.response.bits             := Mux(word, Cat(Fill(32, result(31)), result(31, 0)), result)
  when(io.response.fire)(state := idle)

  val r              = io.request.bits
  val signedDivision = !r.operation(0)
  val signedA        = Mux(r.operation(2), signedDivision, r.operation === 1.U || r.operation === 2.U)
  val signedB        = Mux(r.operation(2), signedDivision, r.operation === 1.U)
  val a              = Mux(r.word, Cat(Fill(32, signedA && r.a(31)), r.a(31, 0)), r.a)
  val rhs            = Mux(r.word, Cat(Fill(32, signedB && r.rhs(31)), r.rhs(31, 0)), r.rhs)
  val negativeA      = signedA && a(63)
  val negativeB      = signedB && rhs(63)
  val magnitudeA     = Mux(negativeA, -a, a)
  val magnitudeB     = Mux(negativeB, -rhs, rhs)
  val product        = (Cat(negativeA, a).asSInt * Cat(negativeB, rhs).asSInt).asUInt

  when(io.request.fire) {
    word := r.word
    when(!r.operation(2)) {
      result := Mux(r.operation === 0.U, product(63, 0), product(127, 64))
      state  := respond
    }.otherwise {
      quotient          := magnitudeA
      remainder         := 0.U
      divisor           := magnitudeB
      count             := 64.U
      negateQuotient    := negativeA ^ negativeB
      negateRemainder   := negativeA
      returnRemainder   := r.operation(1)
      when(rhs === 0.U) {
        result := Mux(r.operation(1), a, ~0.U(64.W))
        state  := respond
      }.otherwise(state := divide)
    }
  }

  val shifted       = Cat(remainder(63, 0), quotient(63))
  val subtract      = shifted >= divisor
  val nextRemainder = Mux(subtract, shifted - divisor, shifted)
  val nextQuotient  = Cat(quotient(62, 0), subtract)
  when(state === divide) {
    remainder := nextRemainder
    quotient  := nextQuotient
    count     := count - 1.U
    when(count === 1.U) {
      result := Mux(
        returnRemainder,
        Mux(negateRemainder, -nextRemainder(63, 0), nextRemainder(63, 0)),
        Mux(negateQuotient, -nextQuotient, nextQuotient)
      )
      state  := respond
    }
  }
}
