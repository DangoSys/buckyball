package examples.balls.int2fp

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.int2fp.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.int2fp.configs.Int2FpBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class Int2FpBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = Int2FpBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val unit: Instance[Int2Fp] =
    Instantiate(new Int2Fp(fp, mp, bp, ball, ballParam))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  for (i <- 0 until ball.inBW) unit.io.bankRead(i) <> blink.bankRead(i)
  for (i <- 0 until ball.outBW) unit.io.bankWrite(i) <> blink.bankWrite(i)
  blink.status <> unit.io.status

}

object Int2FpBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Int2FpBall(fp, mp, bp, id)).blink, None)

}
