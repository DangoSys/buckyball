package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

/** Element arithmetic shared by the e8/e16/e32 integer vector instructions. */
@instantiable
class IALU extends Module {

  @public
  val io = IO(new Bundle {
    val a             = Input(UInt(32.W))
    val b             = Input(UInt(32.W))
    val c             = Input(UInt(32.W))
    val carry         = Input(Bool())
    val vxrm          = Input(UInt(2.W))
    val selector      = Input(UInt(5.W))
    val sew           = Input(UInt(2.W))
    val funct6        = Input(UInt(6.W))
    val multiplyClass = Input(Bool())
    val result        = Output(UInt(32.W))
    val saturated     = Output(Bool())
    val legal         = Output(Bool())
  })

  val mask = MuxLookup(io.sew, "hffffffff".U)(Seq(0.U -> "hff".U, 1.U -> "hffff".U))
  val a    = io.a & mask
  val b    = io.b & mask
  val c    = io.c & mask

  val sa = MuxLookup(io.sew, a)(Seq(
    0.U -> Cat(Fill(24, a(7)), a(7, 0)),
    1.U -> Cat(Fill(16, a(15)), a(15, 0))
  )).asSInt

  val sb = MuxLookup(io.sew, b)(Seq(
    0.U -> Cat(Fill(24, b(7)), b(7, 0)),
    1.U -> Cat(Fill(16, b(15)), b(15, 0))
  )).asSInt

  val width               = 8.U(6.W) << io.sew
  val shift               = b(4, 0) & (width - 1.U)
  val signedProduct       = sa * sb
  val unsignedProduct     = a * b
  val mixedProduct        = sa * Cat(0.U(1.W), b).asSInt
  val reverseMixedProduct = Cat(0.U(1.W), a).asSInt * sb
  val unsignedSum         = a +& b
  val signedSum           = sa +& sb
  val signedDifference    = sa -& sb
  val maximum             = Cat(0.U(1.W), mask(31, 1)).asSInt
  val minimum             = -(maximum +& 1.S)
  val signedOverflow      = signedSum > maximum || signedSum < minimum
  val differenceOverflow  = signedDifference > maximum || signedDifference < minimum
  val carrySum            = unsignedSum + io.carry
  val borrow              = a < (b +& io.carry)
  val wideMask            = Mux(io.sew === 0.U, "hffff".U, "hffffffff".U)
  val wideA               = io.a & wideMask
  val signedWideA         = Mux(io.sew === 0.U, Cat(Fill(16, io.a(15)), io.a(15, 0)), io.a).asSInt
  val narrowShift         = io.b(4, 0) & ((width << 1) - 1.U)
  val wideFirst           = Mux(io.funct6(2), wideA, a)
  val signedWideFirst     = Mux(io.funct6(2), signedWideA, sa)
  val extensionByte       = io.selector(2, 1) === 2.U || io.sew === 1.U
  val extensionUnsigned   = Mux(extensionByte, io.a(7, 0), io.a(15, 0))
  val extensionSigned     = Mux(extensionByte, Cat(Fill(24, io.a(7)), io.a(7, 0)), Cat(Fill(16, io.a(15)), io.a(15, 0)))

  // All fixed-point operations share one rounding path. A signed container
  // also holds unsigned inputs, with an extra zero sign bit.
  val roundInput  = WireDefault(0.S(65.W))
  val roundShift  = WireDefault(0.U(6.W))
  when(io.multiplyClass) {
    roundShift := 1.U
    switch(io.funct6) {
      is(8.U)(roundInput  := unsignedSum.zext)
      is(9.U)(roundInput  := signedSum)
      is(10.U)(roundInput := a.zext - b.zext)
      is(11.U)(roundInput := signedDifference)
    }
  }.otherwise {
    roundInput := Mux(io.funct6(0), sa.pad(65), a.zext.pad(65))
    roundShift := shift
    when(io.funct6 === 39.U) {
      roundInput := signedProduct
      roundShift := width - 1.U
    }
    when(io.funct6 === 46.U || io.funct6 === 47.U) {
      roundInput := Mux(io.funct6(0), signedWideA.pad(65), wideA.zext.pad(65))
      roundShift := narrowShift
    }
  }
  val shifted     = roundInput >> roundShift
  val discardMask = ((1.U(65.W) << roundShift) - 1.U)(64, 0)
  val discarded   = (roundInput.asUInt & discardMask).orR
  val halfway     = roundShift =/= 0.U && (roundInput.asUInt >> (roundShift - 1.U))(0)
  val belowHalf   = (roundInput.asUInt & (discardMask >> 1)).orR

  val increment = MuxLookup(io.vxrm, false.B)(Seq(
    0.U -> halfway,
    1.U -> (halfway && (belowHalf || shifted.asUInt(0))),
    3.U -> (!shifted.asUInt(0) && discarded)
  ))

  val rounded         = shifted +& increment.asUInt.zext
  val roundedOverflow = rounded > maximum || rounded < minimum

  io.result                   := 0.U
  io.saturated                := false.B
  io.legal                    := true.B
  when(io.multiplyClass) {
    switch(io.funct6) {
      is(8.U, 9.U, 10.U, 11.U)(io.result := rounded.asUInt & mask)
      is(18.U)(io.result                 := Mux(io.selector(0), extensionSigned, extensionUnsigned) & mask)
      is(32.U)(io.result                 := Mux(b === 0.U, mask, a / b))
      is(33.U)(io.result                 := Mux(b === 0.U, mask, (sa / sb).asUInt) & mask)
      is(34.U)(io.result                 := Mux(b === 0.U, a, a % b))
      is(35.U)(io.result                 := Mux(b === 0.U, a, (sa.pad(33) % sb.pad(33)).asUInt(31, 0)) & mask)
      is(36.U)(io.result                 := (unsignedProduct >> width) & mask)
      is(37.U)(io.result                 := unsignedProduct & mask)
      is(38.U)(io.result                 := (mixedProduct.asUInt >> width) & mask)
      is(39.U)(io.result                 := (signedProduct.asUInt >> width) & mask)
      is(41.U)(io.result                 := (b * c + a) & mask)
      is(43.U)(io.result                 := (a - b * c) & mask)
      is(45.U)(io.result                 := (unsignedProduct + c) & mask)
      is(47.U)(io.result                 := (c - unsignedProduct) & mask)
      is(48.U, 52.U)(io.result           := (wideFirst + b) & wideMask)
      is(49.U, 53.U)(io.result           := (signedWideFirst + sb).asUInt & wideMask)
      is(50.U, 54.U)(io.result           := (wideFirst - b) & wideMask)
      is(51.U, 55.U)(io.result           := (signedWideFirst - sb).asUInt & wideMask)
      is(56.U)(io.result                 := unsignedProduct & wideMask)
      is(58.U)(io.result                 := mixedProduct.asUInt & wideMask)
      is(59.U)(io.result                 := signedProduct.asUInt & wideMask)
      is(60.U)(io.result                 := (unsignedProduct + io.c) & wideMask)
      is(61.U)(io.result                 := (signedProduct.asUInt + io.c) & wideMask)
      is(62.U)(io.result                 := (mixedProduct.asUInt + io.c) & wideMask)
      is(63.U)(io.result                 := (reverseMixedProduct.asUInt + io.c) & wideMask)
    }
    io.legal := (io.funct6 >= 8.U && io.funct6 <= 11.U) ||
      (io.funct6 === 18.U &&
        ((io.selector === 4.U || io.selector === 5.U) && io.sew === 2.U ||
          (io.selector === 6.U || io.selector === 7.U) && io.sew >= 1.U)) ||
      (io.funct6 >= 32.U && io.funct6 <= 39.U) ||
      Seq(41, 43, 45, 47).map(n => io.funct6 === n.U).reduce(_ || _) ||
      (io.sew < 2.U && io.funct6 >= 48.U && io.funct6 =/= 57.U)
  }.otherwise {
    switch(io.funct6) {
      is(0.U)(io.result        := (a + b) & mask)
      is(2.U)(io.result        := (a - b) & mask)
      is(3.U)(io.result        := (b - a) & mask)
      is(4.U)(io.result        := Mux(a < b, a, b))
      is(5.U)(io.result        := Mux(sa < sb, a, b))
      is(6.U)(io.result        := Mux(a > b, a, b))
      is(7.U)(io.result        := Mux(sa > sb, a, b))
      is(9.U)(io.result        := a & b)
      is(10.U)(io.result       := a | b)
      is(11.U)(io.result       := a ^ b)
      is(16.U)(io.result       := carrySum & mask)
      is(17.U)(io.result       := (carrySum >> width)(0))
      is(18.U)(io.result       := (a - b - io.carry) & mask)
      is(19.U)(io.result       := borrow)
      is(23.U)(io.result       := Mux(io.carry, b, a))
      is(24.U)(io.result       := a === b)
      is(25.U)(io.result       := a =/= b)
      is(26.U)(io.result       := a < b)
      is(27.U)(io.result       := sa < sb)
      is(28.U)(io.result       := a <= b)
      is(29.U)(io.result       := sa <= sb)
      is(30.U)(io.result       := a > b)
      is(31.U)(io.result       := sa > sb)
      is(32.U) { io.result := Mux(unsignedSum > mask, mask, unsignedSum); io.saturated := unsignedSum > mask }
      is(33.U) {
        io.result    := Mux(signedOverflow, Mux(sa < 0.S, minimum, maximum), signedSum).asUInt & mask
        io.saturated := signedOverflow
      }
      is(34.U) { io.result := Mux(a < b, 0.U, a - b); io.saturated := a < b }
      is(35.U) {
        io.result    := Mux(differenceOverflow, Mux(sa < 0.S, minimum, maximum), signedDifference).asUInt & mask
        io.saturated := differenceOverflow
      }
      is(37.U)(io.result       := (a << shift) & mask)
      is(39.U) {
        io.result    := Mux(roundedOverflow, Mux(rounded < 0.S, minimum, maximum), rounded).asUInt & mask
        io.saturated := roundedOverflow
      }
      is(40.U)(io.result       := a >> shift)
      is(41.U)(io.result       := (sa.pad(33) >> shift).asUInt(31, 0) & mask)
      is(42.U, 43.U)(io.result := rounded.asUInt & mask)
      is(44.U)(io.result       := (wideA >> narrowShift) & mask)
      is(45.U)(io.result       := (signedWideA >> narrowShift).asUInt & mask)
      is(46.U) {
        io.result    := Mux(rounded.asUInt > mask, mask, rounded.asUInt)
        io.saturated := rounded.asUInt > mask
      }
      is(47.U) {
        io.result    := Mux(roundedOverflow, Mux(rounded < 0.S, minimum, maximum), rounded).asUInt & mask
        io.saturated := roundedOverflow
      }
    }
    io.legal := Seq(0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
      37, 39, 40, 41, 42, 43)
      .map(n => io.funct6 === n.U).reduce(_ || _) ||
      (io.sew < 2.U && io.funct6 >= 44.U && io.funct6 <= 47.U)
  }
  when(io.sew > 2.U)(io.legal := false.B)
}
