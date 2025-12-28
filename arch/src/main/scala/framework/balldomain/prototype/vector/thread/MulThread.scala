package framework.balldomain.prototype.vector.thread

import chisel3._
import chisel3.util._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import framework.balldomain.prototype.vector.bond.VVV
import framework.balldomain.prototype.vector.op.MulOp

class MulThread(config: VectorBallParam) extends BaseThread(config, "mul") {
  val mulOp   = Module(new MulOp(this.config))
  val vvvBond = IO(new VVV(this.config, this.config.inputWidth, this.config.outputWidth))

  // Connect MulOp and VVVBond
  mulOp.io.in <> vvvBond.in
  mulOp.io.out <> vvvBond.out
}
