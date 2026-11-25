package examples.balls.smatmul

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.smatmul.{Params => ParametersProto}

import examples.balls.smatmul.configs.SMatMulBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class SMatMulBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = SMatMulBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(link, route))

  val smatmulArrayUnit: Instance[SMatMulUnit] =
    Instantiate(new SMatMulUnit(link, ball, ballParam))

  smatmulArrayUnit.io.cmdReq <> blink.cmdReq
  smatmulArrayUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    smatmulArrayUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  for (i <- 0 until outBW) {
    smatmulArrayUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  blink.status <> smatmulArrayUnit.io.status

  // MMIO: this Ball does not consume MMIO metadata
}

object SMatMulBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new SMatMulBall(link, route)).blink, None)

}
