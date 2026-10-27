package examples.balls.smatmul

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.smatmul.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.smatmul.configs.SMatMulBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class SMatMulBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = SMatMulBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val smatmulArrayUnit: Instance[SMatMulUnit] =
    Instantiate(new SMatMulUnit(fp, mp, bp, ball, ballParam))

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
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new SMatMulBall(fp, mp, bp, id)).blink, None)

}
