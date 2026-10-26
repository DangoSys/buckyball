package examples.balls.vector.thread

import chisel3.experimental.hierarchy.public

import chisel3.experimental.hierarchy.instantiable

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import examples.balls.vector.configs.VectorBallParam
import examples.balls.vector.bond.VVV
import examples.balls.vector.op.CascadeOp

@instantiable
class CasThread(val ballParam: VectorBallParam)
    extends BaseThread(
      "cascade",
      ballParam
    ) {
  val cascadeOp = Instantiate(new CascadeOp(ballParam))

  @public
  val vvvBond = IO(
    new VVV(
      this.config.outputWidth,
      this.config.outputWidth,
      ballParam
    )
  )

  // Connect CascadeOp and VVVBond
  cascadeOp.io.in <> vvvBond.in
  cascadeOp.io.out <> vvvBond.out
}
