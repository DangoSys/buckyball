package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class DivRemRequest(val maxWidth: Int) extends Bundle {
  val a         = UInt(maxWidth.W)
  val b         = UInt(maxWidth.W)
  val sew       = UInt(2.W)
  val signed    = Bool()
  val remainder = Bool()
}

@instantiable
class DivRem(val maxWidth: Int) extends Module {
  require(maxWidth == 32 || maxWidth == 64)

  @public val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new DivRemRequest(maxWidth)))
    val response = Decoupled(UInt(maxWidth.W))
    val clear    = Input(Bool())
  })

  val idle :: iterate :: complete :: Nil = Enum(3)
  val state                              = RegInit(idle)
  val quotient                           = Reg(UInt(maxWidth.W))
  val divisor                            = Reg(UInt(maxWidth.W))
  val rest                               = Reg(UInt((maxWidth + 1).W))
  val maskSaved                          = Reg(UInt(maxWidth.W))
  val count                              = Reg(UInt(7.W))
  val quotientNegative                   = Reg(Bool())
  val restNegative                       = Reg(Bool())
  val selectRest                         = Reg(Bool())
  val result                             = Reg(UInt(maxWidth.W))
  val width                              = 8.U(7.W) << io.request.bits.sew
  val mask                               = Fill(maxWidth, 1.U(1.W)) >> (maxWidth.U - width)
  val a                                  = io.request.bits.a & mask
  val b                                  = io.request.bits.b & mask
  val signA                              = io.request.bits.signed && (a >> (width - 1.U))(0)
  val signB                              = io.request.bits.signed && (b >> (width - 1.U))(0)
  val magnitudeA                         = Mux(signA, -a & mask, a)
  val magnitudeB                         = Mux(signB, -b & mask, b)
  val minimum                            = (1.U(maxWidth.W) << (width - 1.U))(maxWidth - 1, 0)
  val overflow                           = io.request.bits.signed && a === minimum && b === mask
  val shifted                            = Cat(rest(maxWidth - 1, 0), quotient(maxWidth - 1))
  val fits                               = shifted >= Cat(0.U(1.W), divisor)
  val nextRest                           = Mux(fits, shifted - divisor, shifted)
  val nextQuotient                       = Cat(quotient(maxWidth - 2, 0), fits)
  val signedQuotient                     = Mux(quotientNegative, -nextQuotient, nextQuotient) & maskSaved
  val signedRest                         = Mux(restNegative, -nextRest(maxWidth - 1, 0), nextRest(maxWidth - 1, 0)) & maskSaved
  io.request.ready             := state === idle && !io.clear
  io.response.valid            := state === complete && !io.clear
  io.response.bits             := result
  when(io.request.fire) {
    assert(width <= maxWidth.U)
    maskSaved        := mask
    quotientNegative := signA ^ signB
    restNegative     := signA
    selectRest       := io.request.bits.remainder
    when(b === 0.U) {
      result := Mux(io.request.bits.remainder, a, mask)
      state  := complete
    }.elsewhen(overflow) {
      result := Mux(io.request.bits.remainder, 0.U, minimum)
      state  := complete
    }.otherwise {
      quotient := (magnitudeA << (maxWidth.U - width))(maxWidth - 1, 0)
      divisor  := magnitudeB
      rest     := 0.U
      count    := width
      state    := iterate
    }
  }
  when(state === iterate) {
    quotient := nextQuotient
    rest     := nextRest
    count    := count - 1.U
    when(count === 1.U) {
      result := Mux(selectRest, signedRest, signedQuotient)
      state  := complete
    }
  }
  when(io.response.fire)(state := idle)
  when(io.clear)(state         := idle)
}
