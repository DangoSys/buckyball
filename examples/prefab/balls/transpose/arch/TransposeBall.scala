package examples.balls.transpose

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.transpose.{Params => ParametersProto}

import examples.balls.transpose.configs.TransposeBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import examples.balls.transpose.Transpose

@instantiable
class TransposeBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = TransposeBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(link, route))

  val transposeUnit: Instance[Transpose] =
    Instantiate(new Transpose(link, ball, ballParam))

  transposeUnit.io.cmdReq <> blink.cmdReq
  transposeUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    transposeUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  for (i <- 0 until outBW) {
    transposeUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  blink.status <> transposeUnit.io.status

  // MMIO: this Ball does not consume MMIO metadata
}

object TransposeBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new TransposeBall(link, route)).blink, None)

}
