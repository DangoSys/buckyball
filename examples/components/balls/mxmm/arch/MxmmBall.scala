package examples.balls.mxmm

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxmm.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.mxmm.configs.MxmmBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class MxmmBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = MxmmBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val unit =
    Instantiate(new Unit(fp, mp, bp, ball, ballParam))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  unit.io.bankRead <> blink.bankRead
  unit.io.bankWrite <> blink.bankWrite
  blink.status <> unit.io.status
}

object MxmmBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new MxmmBall(fp, mp, bp, id)).blink, None)

}
