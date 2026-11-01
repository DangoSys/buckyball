package examples.balls.mxint8matmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

/**
 * One C element. `mac` adds `lanes` INT8 code products into the exact block sum p (|p| <= 32 * 128 * 128 = 2^19);
 * `convert` turns p * 2^scale into an FP32 term in integer form; `accumulate` adds the term into the FP32 accumulator
 * (round to nearest even).
 */
@instantiable
class PE(lanes: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val active     = Input(Bool())
    val mac        = Input(Bool())
    val first      = Input(Bool())
    val a          = Input(UInt((lanes * 8).W))
    val b          = Input(UInt((lanes * 8).W))
    val convert    = Input(Bool())
    val scale      = Input(SInt(10.W))
    val accumulate = Input(Bool())
    val load       = Input(Bool())
    val init       = Input(UInt(32.W))
    val out        = Output(UInt(32.W))
  })

  val sum         = Reg(SInt(21.W))
  val term        = Reg(UInt(32.W))
  val accumulator = Reg(UInt(33.W))

  val products = (0 until lanes).map(l => io.a(8 * l + 7, 8 * l).asSInt * io.b(8 * l + 7, 8 * l).asSInt)
  when(io.mac)(sum := Mux(io.first, 0.S, sum) + products.reduce(_ +& _))

  val sign      = sum < 0.S
  val magnitude = Mux(sign, (0.S - sum).asUInt, sum.asUInt)(19, 0)
  val msb       = Log2(magnitude)
  val exponent  = msb.zext + io.scale + 127.S(12.W)
  val normal    = Cat(sign, exponent.asUInt(7, 0), (magnitude << (23.U - msb))(22, 0))
  // Below the normal range the value is |p| * 2^(scale + 149) subnormal units.
  val position  = io.scale + 149.S(12.W)
  val distance  = (0.S - position).asUInt
  val extended  = Cat(magnitude, 0.U(20.W)) >> distance(4, 0)
  val quotient  = extended(39, 20)
  val rounded   = quotient +& (extended(19) && (extended(18, 0).orR || quotient(0)))

  val units = Mux(
    position >= 0.S,
    (magnitude << position.asUInt(4, 0))(22, 0),
    Mux(distance > 20.U, 0.U, rounded)
  )

  val subnormal = Cat(sign, units.pad(31))
  when(io.convert) {
    term := Mux(magnitude === 0.U, 0.U, Mux(exponent > 0.S, normal, subnormal))
    when(io.active && magnitude =/= 0.U)(assert(exponent < 255.S, "MXINT8MM block term overflows FP32"))
  }

  val adder = Module(new AddRecFN(8, 24))
  adder.io.subOp            := false.B
  adder.io.a                := accumulator
  adder.io.b                := recFNFromFN(8, 24, term)
  adder.io.roundingMode     := consts.round_near_even
  adder.io.detectTininess   := consts.tininess_afterRounding
  when(io.accumulate) {
    accumulator := adder.io.out
    when(io.active)(assert(fNFromRecFN(8, 24, adder.io.out)(30, 23) =/= 255.U, "MXINT8MM accumulation overflows FP32"))
  }
  when(io.load)(accumulator := recFNFromFN(8, 24, io.init))

  io.out := fNFromRecFN(8, 24, accumulator)
}
