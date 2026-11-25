package examples.balls.relu

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.relu.{Params => ParametersProto}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}

@instantiable
class ReluBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  @public
  val blink = IO(new BlinkIO(link, route))

  val unit: Instance[Relu] = Instantiate(new Relu(link, ball))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  unit.io.bankRead(0) <> blink.bankRead(0)
  unit.io.bankWrite(0) <> blink.bankWrite(0)
  blink.status <> unit.io.status
}

object ReluBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new ReluBall(link, route)).blink, None)

}
