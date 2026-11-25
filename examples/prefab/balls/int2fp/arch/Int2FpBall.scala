package examples.balls.int2fp

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.int2fp.{Params => ParametersProto}

import examples.balls.int2fp.configs.Int2FpBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class Int2FpBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = Int2FpBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  @public
  val blink = IO(new BlinkIO(link, route))

  val unit: Instance[Int2Fp] =
    Instantiate(new Int2Fp(link, ball, ballParam))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  for (i <- 0 until ball.inBW) unit.io.bankRead(i) <> blink.bankRead(i)
  for (i <- 0 until ball.outBW) unit.io.bankWrite(i) <> blink.bankWrite(i)
  blink.status <> unit.io.status

}

object Int2FpBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Int2FpBall(link, route)).blink, None)

}
