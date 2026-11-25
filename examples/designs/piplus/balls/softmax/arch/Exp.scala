package examples.designs.piplus.balls.softmax

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import hardfloat._

/**
 * e = 2^((x - m) * log2(e)) of one FP32 lane as workloads/isa/softmax.h specifies it, one FP32 operation per stage,
 * every stage moving when `advance`:
 *
 *   d = x - m | u = d * log2(e) | n = floor(u), zero = u < -126 | f = u - n | P(f), six Horner steps | e = P(f) * 2^n
 *
 * Values are recoded FP32 (33 bits).
 */
@instantiable
class Exp extends Module {

  @public
  val io = IO(new Bundle {
    val advance  = Input(Bool())
    val inValid  = Input(Bool())
    val inMask   = Input(Bool())
    val x        = Input(UInt(33.W))
    val m        = Input(UInt(33.W))
    val outValid = Output(Bool())
    val outMask  = Output(Bool())
    val e        = Output(UInt(33.W))
  })

  val rne   = consts.round_near_even
  val after = consts.tininess_afterRounding

  // d = x - m
  val sub = Module(new AddRecFN(8, 24))
  sub.io.subOp          := true.B
  sub.io.a              := io.x
  sub.io.b              := io.m
  sub.io.roundingMode   := rne
  sub.io.detectTininess := after
  val valid1 = RegEnable(io.inValid, false.B, io.advance)
  val mask1  = RegEnable(io.inMask, io.advance)
  val d1     = RegEnable(sub.io.out, io.advance)

  // u = d * log2(e)
  val scale = Module(new MulRecFN(8, 24))
  scale.io.a              := d1
  scale.io.b              := recFNFromFN(8, 24, "h3fb8aa3b".U(32.W))
  scale.io.roundingMode   := rne
  scale.io.detectTininess := after
  val valid2 = RegEnable(valid1, false.B, io.advance)
  val mask2  = RegEnable(mask1, io.advance)
  val u2     = RegEnable(scale.io.out, io.advance)

  // n = floor(u); u < -126 gives e = 0
  val floor = Module(new RecFNToIN(8, 24, 32))
  floor.io.in           := u2
  floor.io.roundingMode := consts.round_min
  floor.io.signedOut    := true.B
  val below = Module(new CompareRecFN(8, 24))
  below.io.a         := u2
  below.io.b         := recFNFromFN(8, 24, "hc2fc0000".U(32.W))
  below.io.signaling := false.B
  val valid3 = RegEnable(valid2, false.B, io.advance)
  val mask3  = RegEnable(mask2, io.advance)
  val u3     = RegEnable(u2, io.advance)
  val n3     = RegEnable(floor.io.out, io.advance)
  val zero3  = RegEnable(below.io.lt, io.advance)

  // f = u - n
  val toFloat = Module(new INToRecFN(32, 8, 24))
  toFloat.io.signedIn       := true.B
  toFloat.io.in             := n3
  toFloat.io.roundingMode   := rne
  toFloat.io.detectTininess := after
  val fraction = Module(new AddRecFN(8, 24))
  fraction.io.subOp          := true.B
  fraction.io.a              := u3
  fraction.io.b              := toFloat.io.out
  fraction.io.roundingMode   := rne
  fraction.io.detectTininess := after
  val valid4 = RegEnable(valid3, false.B, io.advance)
  val mask4  = RegEnable(mask3, io.advance)
  val f4     = RegEnable(fraction.io.out, io.advance)
  val n4     = RegEnable(n3(7, 0), io.advance)
  val zero4  = RegEnable(zero3, io.advance)

  // P(f) = ((((((c6 f + c5) f + c4) f + c3) f + c2) f + c1) f + 1)
  val steps = Seq(0x3aaec3ffL, 0x3c1d955bL, 0x3d635847L, 0x3e75fdf0L, 0x3f317218L, 0x3f800000L)
    .map(c => Instantiate(new Horner(c)))
  steps.head.io.inValid := valid4
  steps.head.io.inMask  := mask4
  steps.head.io.inP     := recFNFromFN(8, 24, "h39218489".U(32.W))
  steps.head.io.inF     := f4
  steps.head.io.inN     := n4
  steps.head.io.inZero  := zero4
  for ((step, next) <- steps.zip(steps.tail)) {
    next.io.inValid := step.io.outValid
    next.io.inMask  := step.io.outMask
    next.io.inP     := step.io.outP
    next.io.inF     := step.io.outF
    next.io.inN     := step.io.outN
    next.io.inZero  := step.io.outZero
  }
  steps.foreach(_.io.advance := io.advance)
  val last = steps.last

  // e = P(f) * 2^n, exact: 2^n is normal for n >= -126 and P(f) is in [1, 2)
  val power = Module(new MulRecFN(8, 24))
  power.io.a              := last.io.outP
  power.io.b              := recFNFromFN(8, 24, Cat(0.U(1.W), last.io.outN + 127.U, 0.U(23.W)))
  power.io.roundingMode   := rne
  power.io.detectTininess := after

  io.outValid := RegEnable(last.io.outValid, false.B, io.advance)
  io.outMask  := RegEnable(last.io.outMask, io.advance)
  io.e        := RegEnable(Mux(last.io.outZero, 0.U(33.W), power.io.out), io.advance)
}
