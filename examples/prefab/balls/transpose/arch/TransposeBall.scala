package examples.balls.transpose

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.transpose.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.transpose.configs.TransposeBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}
import examples.balls.transpose.Transpose

@instantiable
class TransposeBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = TransposeBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val transposeUnit: Instance[Transpose] =
    Instantiate(new Transpose(fp, mp, bp, ball, ballParam))

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
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new TransposeBall(fp, mp, bp, id)).blink, None)

}
