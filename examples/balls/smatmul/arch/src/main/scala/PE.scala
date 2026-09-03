package examples.balls.smatmul

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class PE extends Module {

  @public
  val io = IO(new Bundle {
    val aIn       = Input(SInt(8.W))
    val aInValid  = Input(Bool())
    val bIn       = Input(SInt(8.W))
    val bInValid  = Input(Bool())
    val clear     = Input(Bool())
    val aOut      = Output(SInt(8.W))
    val aOutValid = Output(Bool())
    val bOut      = Output(SInt(8.W))
    val bOutValid = Output(Bool())
    val sum       = Output(SInt(32.W))
  })

  val accumulator = RegInit(0.S(32.W))
  val aPipe       = RegInit(0.S(8.W))
  val aPipeValid  = RegInit(false.B)
  val bPipe       = RegInit(0.S(8.W))
  val bPipeValid  = RegInit(false.B)

  when(io.clear) {
    accumulator := 0.S
    aPipeValid  := false.B
    bPipeValid  := false.B
  }.elsewhen(aPipeValid && bPipeValid) {
    accumulator := accumulator + (aPipe * bPipe)
  }
  when(!io.clear) {
    aPipe      := io.aIn
    aPipeValid := io.aInValid
    bPipe      := io.bIn
    bPipeValid := io.bInValid
  }

  io.aOut      := aPipe
  io.aOutValid := aPipeValid
  io.bOut      := bPipe
  io.bOutValid := bPipeValid
  io.sum       := accumulator
}
