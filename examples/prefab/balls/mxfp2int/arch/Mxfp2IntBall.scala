package examples.balls.mxfp2int

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxfp2int.{Params => ParametersProto}

import examples.balls.mxfp2int.configs.Mxfp2IntBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.ball.blink.{BlinkIO, HasBlink}

/**
 * Mxfp2IntBall - MXFP4 to INT8 dequantizer.
 *
 * Wraps PipelinedMxfp2Int and connects to the Blink protocol; the per-block
 * E8M0 scales come from the second source bank.
 */
@instantiable
class Mxfp2IntBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = Mxfp2IntBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  require(ballCommonConfig.mmioReadBW == 0 && ballCommonConfig.mmioWriteBW == 0)

  @public
  val blink = IO(new BlinkIO(link, route))

  val mxfp2intUnit: Instance[PipelinedMxfp2Int] = Instantiate(
    new PipelinedMxfp2Int(link, ball, ballParam)
  )

  mxfp2intUnit.io.cmdReq <> blink.cmdReq
  mxfp2intUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    mxfp2intUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  for (i <- 0 until outBW) {
    mxfp2intUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  blink.status <> mxfp2intUnit.io.status
}

object Mxfp2IntBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxfp2IntBall(link, route)).blink, None)

}
