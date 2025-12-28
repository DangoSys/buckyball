package framework.balldomain.prototype.vector.thread

import chisel3._
import chisel3.util._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import framework.balldomain.prototype.vector.bond.VVV
import framework.balldomain.prototype.vector.op.CascadeOp

class CasThread(config: VectorBallParam) extends BaseThread(config, "cascade") {
  val cascadeOp = Module(new CascadeOp(this.config))
  val vvvBond   = IO(new VVV(this.config, this.config.outputWidth, this.config.outputWidth))

  // Connect CascadeOp and VVVBond
  cascadeOp.io.in <> vvvBond.in
  cascadeOp.io.out <> vvvBond.out
}
