package framework.balldomain.prototype.vector.op

import chisel3._
import chisel3.util._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import framework.balldomain.prototype.vector.bond.VVV

class MulOp(val config: VectorBallParam) extends Module {
  val lane        = config.lane
  val inputWidth  = config.inputWidth
  val outputWidth = config.outputWidth

  val io     = IO(new VVV(config, inputWidth, outputWidth))
  val reg1   = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
  val reg2   = RegInit(VecInit(Seq.fill(lane)(0.U(inputWidth.W))))
  val cnt    = RegInit(0.U(log2Ceil(lane).W))
  val active = RegInit(false.B)

  io.out.valid := active && io.out.ready
  io.in.ready  := io.out.ready

  when(io.in.valid) {
    reg1   := io.in.bits.in1
    reg2   := io.in.bits.in2
    cnt    := 0.U
    active := true.B
  }.elsewhen(active && io.out.ready) {
    cnt := cnt + 1.U
    when(cnt === (lane - 1).U) {
      active := false.B
    }
  }

  for (i <- 0 until lane) {
    io.out.bits.out(i) := Mux(io.out.valid, reg1(cnt) * reg2(i), 0.U)
  }
}
