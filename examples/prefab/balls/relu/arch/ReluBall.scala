package examples.balls.relu

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.relu.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}

@instantiable
class ReluBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val unit: Instance[Relu] = Instantiate(new Relu(fp, mp, bp, ball))
  unit.io.cmdReq <> blink.cmdReq
  unit.io.cmdResp <> blink.cmdResp
  unit.io.bankRead(0) <> blink.bankRead(0)
  unit.io.bankWrite(0) <> blink.bankWrite(0)
  blink.status <> unit.io.status
}

object ReluBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new ReluBall(fp, mp, bp, id)).blink, None)

}
