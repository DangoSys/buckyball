package examples.balls.vector.thread

import chisel3._
import chisel3.util._
import examples.balls.vector.configs.VectorBallParam
import examples.balls.vector.bond.VVV
import examples.balls.vector.op.CascadeOp

class CasThread(config: VectorBallParam) extends BaseThread(config, "cascade") {
  val cascadeOp = Module(new CascadeOp(this.config))
  val vvvBond   = IO(new VVV(this.config, this.config.outputWidth, this.config.outputWidth))

  // Connect CascadeOp and VVVBond
  cascadeOp.io.in <> vvvBond.in
  cascadeOp.io.out <> vvvBond.out
}
