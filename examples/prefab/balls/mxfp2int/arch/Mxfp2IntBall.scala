package examples.balls.mxfp2int

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxfp2int.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.mxfp2int.configs.Mxfp2IntBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}

/**
 * Mxfp2IntBall - MXFP4 to INT8 dequantizer.
 *
 * Wraps PipelinedMxfp2Int and connects to the Blink protocol; the per-block
 * E8M0 scales come from the second source bank.
 */
@instantiable
class Mxfp2IntBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = Mxfp2IntBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  require(ballCommonConfig.mmioReadBW == 0 && ballCommonConfig.mmioWriteBW == 0)

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val mxfp2intUnit: Instance[PipelinedMxfp2Int] = Instantiate(
    new PipelinedMxfp2Int(fp, mp, bp, ball, ballParam)
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
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Mxfp2IntBall(fp, mp, bp, id)).blink, None)

}
