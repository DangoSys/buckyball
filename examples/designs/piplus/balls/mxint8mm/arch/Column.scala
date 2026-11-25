package examples.designs.piplus.balls.mxint8mm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

/**
 * One column of C, a row of the tile at a time. `mac` adds `lanes` INT8 code products into the exact block sum p of
 * the row's element; `convert` turns p * 2^scale into an FP32 term; `out` is x + y in FP32 (round to nearest even):
 * the term into the element, or the bias into its start value.
 */
@instantiable
class Column(lanes: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val mac     = Input(Bool())
    val first   = Input(Bool())
    val a       = Input(UInt((lanes * 8).W))
    val b       = Input(UInt((lanes * 8).W))
    val convert = Input(Bool())
    val active  = Input(Bool())
    val scale   = Input(SInt(10.W))
    val term    = Output(UInt(32.W))
    val x       = Input(UInt(32.W))
    val y       = Input(UInt(32.W))
    val out     = Output(UInt(32.W))
  })

  // |p| reaches 2^19 when every code pair is (-128) * (-128).
  val sum  = Reg(SInt(21.W))
  val term = Reg(UInt(32.W))

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
  io.term := term

  val adder = Module(new AddRecFN(8, 24))
  adder.io.subOp          := false.B
  adder.io.a              := recFNFromFN(8, 24, io.x)
  adder.io.b              := recFNFromFN(8, 24, io.y)
  adder.io.roundingMode   := consts.round_near_even
  adder.io.detectTininess := consts.tininess_afterRounding
  io.out                  := fNFromRecFN(8, 24, adder.io.out)
}
