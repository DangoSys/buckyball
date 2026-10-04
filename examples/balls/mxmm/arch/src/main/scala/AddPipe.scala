package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

@instantiable
class AddPipe extends Module {

  @public val io = IO(new Bundle {
    val valid    = Input(Bool())
    val a        = Input(UInt(33.W))
    val b        = Input(UInt(33.W))
    val outValid = Output(Bool())
    val out      = Output(UInt(33.W))
  })

  val decodedA     = RegEnable(rawFloatFromRecFN(8, 24, io.a), io.valid)
  val decodedB     = RegEnable(rawFloatFromRecFN(8, 24, io.b), io.valid)
  val decodedValid = RegNext(io.valid, false.B)
  val add          = Module(new AddRawFN(8, 24))
  add.io.subOp        := false.B
  add.io.a            := decodedA
  add.io.b            := decodedB
  add.io.roundingMode := consts.round_near_even

  val sum      = RegEnable(add.io.rawOut, decodedValid)
  val invalid  = RegEnable(add.io.invalidExc, decodedValid)
  val sumValid = RegNext(decodedValid, false.B)
  val round    = Module(new RoundRawFNToRecFN(8, 24, 0))
  round.io.in             := sum
  round.io.invalidExc     := invalid
  round.io.infiniteExc    := false.B
  round.io.roundingMode   := consts.round_near_even
  round.io.detectTininess := consts.tininess_afterRounding

  io.out      := RegEnable(round.io.out, sumValid)
  io.outValid := RegNext(sumValid, false.B)
}
