package examples.balls.vector

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.vector.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.vector.configs.VectorBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}
import examples.balls.vector.VecUnit

/**
 * VecBall
 */
@instantiable
class VecBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = VectorBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val inBW  = ball.inBW
  val outBW = ball.outBW

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val vecUnit: Instance[VecUnit] =
    Instantiate(new VecUnit(fp, mp, bp, ball, ballParam))

  vecUnit.io.cmdReq <> blink.cmdReq
  when(blink.cmdReq.fire) {
    val cmd = blink.cmdReq.bits
    assert(cmd.op1_group === 0.U && cmd.op2_group === 0.U && cmd.wr_group === 0.U, "VecBall has no group offsets")
  }
  vecUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    vecUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  for (i <- 0 until outBW) {
    vecUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  blink.status <> vecUnit.io.status

}

object VecBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new VecBall(fp, mp, bp, id)).blink, None)

}
