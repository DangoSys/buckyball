package examples.balls.matadd

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.matadd.{Params => ParametersProto}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}

@instantiable
class MatAddBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  @public
  val blink = IO(new BlinkIO(link, route))

  val unit: Instance[MatAdd] =
    Instantiate(new MatAdd(link, ball))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  unit.io.bankRead(0) <> blink.bankRead(0)
  unit.io.bankRead(1) <> blink.bankRead(1)
  unit.io.bankWrite(0) <> blink.bankWrite(0)
  blink.status <> unit.io.status
}

object MatAddBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new MatAddBall(link, route)).blink, None)

}
