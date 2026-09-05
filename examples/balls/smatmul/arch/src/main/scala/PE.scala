package examples.balls.smatmul

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class PE extends Module {

  // One tile accumulates sixteen signed 8x8 products.  Twenty bits cover the
  // full range [-262144, 258064]; the unit-level accumulator remains 32-bit
  // for reduction across tiles.
  private val tileSumWidth = 20

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

  val accumulator  = RegInit(0.S(tileSumWidth.W))
  val aPipe        = RegInit(0.S(8.W))
  val aPipeValid   = RegInit(false.B)
  val bPipe        = RegInit(0.S(8.W))
  val bPipeValid   = RegInit(false.B)
  val product      = RegInit(0.S(16.W))
  val productValid = RegInit(false.B)

  when(io.clear) {
    accumulator  := 0.S
    aPipeValid   := false.B
    bPipeValid   := false.B
    productValid := false.B
  }.elsewhen(productValid) {
    accumulator := accumulator + product
  }
  when(!io.clear) {
    product      := aPipe * bPipe
    productValid := aPipeValid && bPipeValid
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
  io.sum       := accumulator.pad(32)
}
