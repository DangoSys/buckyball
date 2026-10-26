package examples.balls.vector.op

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util._
import examples.balls.vector.configs.VectorBallParam
import examples.balls.vector.bond.VVV

@instantiable
class CascadeOp(val ballParam: VectorBallParam) extends Module {
  val config = ballParam

  val lane        = config.lane
  val inputWidth  = config.outputWidth // Cascade uses outputWidth as input
  val outputWidth = config.outputWidth

  @public
  val io = IO(new VVV(
    inputWidth,
    outputWidth,
    ballParam
  ))

  val reg1   = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
  val reg2   = RegInit(VecInit(Seq.fill(lane)(0.U(outputWidth.W))))
  val valid1 = RegInit(false.B)
  val valid2 = RegInit(false.B)

  io.in.ready := io.out.ready

  when(io.in.valid) {
    valid1 := true.B
    reg1   := io.in.bits.in1.zip(io.in.bits.in2).map { case (a, b) => a + b }
  }.elsewhen(!io.in.ready) {
    valid1 := valid1
  }.otherwise {
    valid1 := false.B
  }

  val valid = valid1

  when(io.out.ready && valid) {
    io.out.valid    := true.B
    io.out.bits.out := reg1
  }.otherwise {
    io.out.valid    := false.B
    io.out.bits.out := VecInit(Seq.fill(lane)(0.U(outputWidth.W)))
  }
}
