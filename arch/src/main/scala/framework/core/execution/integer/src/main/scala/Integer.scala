// See LICENSE.Berkeley and LICENSE.SiFive for license details.
package framework.core.execution.integer

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.execution.integer.configs.{Params => ModuleParam}

class IntegerIO(b: ModuleParam) extends Bundle {
  val p = b

  val in1       = Input(UInt(p.xLen.W))
  val in2       = Input(UInt(p.xLen.W))
  val fn        = Input(UInt(5.W))
  val dw        = Input(Bool())
  val out       = Output(UInt(p.xLen.W))
  val adder_out = Output(UInt(p.xLen.W))
  val cmp_out   = Output(Bool())

}

@instantiable
class Integer(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new IntegerIO(b))

  val a           = io.in1
  val rhs         = io.in2
  val fn          = io.fn
  val subtract    = fn(3)
  val inverted    = Mux(subtract, ~rhs, rhs)
  val difference  = a ^ inverted
  val conjunction = a & inverted
  val sum         = a + inverted + subtract
  val less        = Mux(a(63) === rhs(63), sum(63), Mux(fn(1), rhs(63), a(63)))
  io.adder_out := sum
  io.cmp_out   := fn(0) ^ Mux(!subtract, difference === 0.U, less)
  val shift        = Cat(rhs(5) && io.dw, rhs(4, 0))
  val high         = Mux(io.dw, a(63, 32), Fill(32, subtract && a(31)))
  val input        = Cat(high, a(31, 0))
  val reverse      = !(fn === 5.U || fn === 11.U || fn === 18.U || fn === 19.U)
  val shiftedInput = Mux(reverse, Reverse(input), input)
  val right        = (Cat(subtract && shiftedInput(63), shiftedInput).asSInt >> shift)(63, 0)
  val left         = Reverse(right)
  val shifts       = Mux(fn === 5.U || fn === 11.U || fn === 19.U, right, 0.U) | Mux(fn === 1.U, left, 0.U)
  val logic        = Mux(fn === 4.U || fn === 6.U || fn === 25.U || fn === 26.U, difference, 0.U) |
    Mux(fn === 6.U || fn === 7.U || fn === 25.U || fn === 24.U, conjunction, 0.U)
  val comparison   = (fn >= 12.U && fn <= 15.U) && less
  val conditional  = Mux((fn === 8.U && rhs.orR) || (fn === 9.U && !rhs.orR), a, 0.U)
  val shiftLogic   = comparison.asUInt | logic | (shifts & Mux(fn === 19.U, 1.U, ~0.U(64.W))) | conditional

  val countInput = MuxLookup(Cat(!io.dw, !rhs(0)), 0.U)(Seq(
    0.U -> a,
    1.U -> Reverse(a),
    2.U -> Cat(1.U(1.W), a(31, 0)),
    3.U -> Cat(1.U(1.W), Reverse(a(31, 0)))
  ))

  val populationInput = Mux(rhs(1), Mux(io.dw, a, a(31, 0)), PriorityEncoderOH(Cat(1.U(1.W), countInput)) - 1.U)(63, 0)
  val count           = PopCount(populationInput)
  val bytes           = a.asTypeOf(Vec(8, UInt(8.W)))

  val unary = MuxLookup(rhs(11, 0), count)(Seq(
    "h287".U -> VecInit(bytes.map(byte => Fill(8, byte.orR))).asUInt,
    "h6b8".U -> VecInit(bytes.reverse).asUInt,
    "h080".U -> a(15, 0),
    "h604".U -> Cat(Fill(56, a(7)), a(7, 0)),
    "h605".U -> Cat(Fill(48, a(15)), a(15, 0))
  ))

  val rotationShift  = Mux(io.dw, 64.U, 32.U) - shift
  val rotationInput  = Mux(fn(0), input, Reverse(input))
  val rotateRight    = (rotationInput >> rotationShift)(63, 0)
  val rotateLeft     = Reverse(rotateRight)
  val rotation       = Mux(fn(0), rotateRight, rotateLeft) | Mux(fn(0), left, right)
  val minimumMaximum = Mux(io.cmp_out, rhs, a)

  val result = MuxLookup(fn, shiftLogic)(Seq(
    0.U  -> sum,
    10.U -> sum,
    16.U -> unary,
    28.U -> minimumMaximum,
    29.U -> minimumMaximum,
    30.U -> minimumMaximum,
    31.U -> minimumMaximum,
    17.U -> rotation,
    18.U -> rotation
  ))

  io.out := Mux(io.dw, result, Cat(Fill(32, result(31)), result(31, 0)))
}
