package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

// c + a * b in FP32, rounded once (round to nearest even), as a fused
// multiply-add of the two elements' FP32 values would give. Two MXFP8
// elements multiply exactly in 8 bits, so only the addition is floating
// point: the exact product and c are added in a wider exponent range and
// rounded once to FP32.
@instantiable
class PE(val latency: Int) extends Module {

  require(latency == 3)

  @public
  val io = IO(new Bundle {
    val valid    = Input(Bool())
    val a        = Input(new Mxfp8Value)
    val b        = Input(new Mxfp8Value)
    val c        = Input(UInt(33.W))
    val out      = Output(UInt(33.W))
    val outValid = Output(Bool())
  })

  // Raw floats of 10 exponent bits hold any product exactly: value =
  // sig / 2^23 * 2^(sExp - 1024), the leading bit of sig at bit 23.
  val expWidth = 10
  val sigWidth = 24
  val bias     = 1 << expWidth

  // Stage 1: the exact product, and c widened.
  val product     = io.a.sig * io.b.sig
  val leading     = 7.U - countLeadingZeros(product)
  val productRaw  = Wire(new RawFloat(expWidth, sigWidth))
  val productZero = product === 0.U
  productRaw.isNaN  := false.B
  productRaw.isInf  := false.B
  productRaw.isZero := productZero
  productRaw.sign   := io.a.sign ^ io.b.sign
  // A zero must not outweigh the other addend when they are aligned.
  productRaw.sExp   := Mux(productZero, 0.S, io.a.exp +& io.b.exp +& leading.zext +& bias.S)
  productRaw.sig    := (product << (23.U - leading))(sigWidth, 0)
  val c    = rawFloatFromRecFN(8, 24, io.c)
  val cRaw = Wire(new RawFloat(expWidth, sigWidth))
  cRaw      := c
  cRaw.sExp := Mux(c.isZero, 0.S, c.sExp +& (bias - (1 << 8)).S)
  val stage1  = Pipe(io.valid, productRaw, 1)
  val stage1C = RegNext(cRaw)

  // Stage 2: the unrounded sum.
  val add = Module(new AddRawFN(expWidth, sigWidth))
  add.io.subOp        := false.B
  add.io.a            := stage1C
  add.io.b            := stage1.bits
  add.io.roundingMode := consts.round_near_even
  val stage2        = Pipe(stage1.valid, add.io.rawOut, 1)
  val stage2Invalid = RegNext(add.io.invalidExc)

  // Stage 3: rounded once to FP32.
  val round = Module(new RoundAnyRawFNToRecFN(expWidth, sigWidth + 2, 8, 24, 0))
  round.io.invalidExc     := stage2Invalid
  round.io.infiniteExc    := false.B
  round.io.in             := stage2.bits
  round.io.roundingMode   := consts.round_near_even
  round.io.detectTininess := consts.tininess_afterRounding
  val stage3 = Pipe(stage2.valid, round.io.out, 1)

  io.outValid := stage3.valid
  io.out      := stage3.bits
  when(io.outValid) {
    assert(fNFromRecFN(8, 24, io.out)(30, 23) =/= 255.U, "MATMUL accumulation must remain finite")
  }
}
