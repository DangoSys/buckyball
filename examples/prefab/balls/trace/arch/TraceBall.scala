package examples.balls.trace

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.trace.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}

/**
 * TraceBall — Debug trace Ball providing cycle counters.
 *
 * The Thread ballISA assigns the funct7 encoding for bdb_counter.
 */
@instantiable
class TraceBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val traceUnit: Instance[Trace] =
    Instantiate(new Trace(fp, mp, bp, ball))

  traceUnit.io.cmdReq <> blink.cmdReq
  traceUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    traceUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  for (i <- 0 until outBW) {
    traceUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  blink.status <> traceUnit.io.status

  // MMIO: this Ball does not consume MMIO metadata
}

object TraceBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new TraceBall(fp, mp, bp, id)).blink, None)

}
