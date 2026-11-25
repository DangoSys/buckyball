package examples.balls.vector

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.vector.{Params => ParametersProto}

import examples.balls.vector.configs.VectorBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}
import examples.balls.vector.VecUnit

/**
 * VecBall
 */
@instantiable
class VecBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = VectorBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val inBW  = ball.inBW
  val outBW = ball.outBW

  @public
  val blink = IO(new BlinkIO(link, route))

  val vecUnit: Instance[VecUnit] =
    Instantiate(new VecUnit(link, ball, ballParam))

  vecUnit.io.cmdReq <> blink.cmdReq
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
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new VecBall(link, route)).blink, None)

}
