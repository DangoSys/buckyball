package examples.balls.vector.thread

import chisel3.experimental.hierarchy.public

import chisel3.experimental.hierarchy.instantiable

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import examples.balls.vector.configs.VectorBallParam
import examples.balls.vector.bond.VVV
import examples.balls.vector.op.MulOp

@instantiable
class MulThread(val ballParam: VectorBallParam)
    extends BaseThread(
      "mul",
      ballParam
    ) {
  val mulOp = Instantiate(new MulOp(ballParam))

  @public
  val vvvBond = IO(
    new VVV(
      this.config.inputWidth,
      this.config.outputWidth,
      ballParam
    )
  )

  // Connect MulOp and VVVBond
  mulOp.io.in <> vvvBond.in
  mulOp.io.out <> vvvBond.out
}
