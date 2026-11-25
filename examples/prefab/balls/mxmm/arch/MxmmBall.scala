package examples.balls.mxmm

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxmm.{Params => ParametersProto}

import examples.balls.mxmm.configs.MxmmBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class MxmmBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = MxmmBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  @public
  val blink = IO(new BlinkIO(link, route))

  val unit =
    Instantiate(new Unit(link, ball, ballParam))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  unit.io.bankRead <> blink.bankRead
  unit.io.bankWrite <> blink.bankWrite
  blink.status <> unit.io.status
}

object MxmmBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new MxmmBall(link, route)).blink, None)

}
