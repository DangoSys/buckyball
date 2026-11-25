package examples.balls.trace

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.trace.{Params => ParametersProto}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}

/**
 * TraceBall — Debug trace Ball providing cycle counters.
 *
 * The Thread ballISA assigns the funct7 encoding for bdb_counter.
 */
@instantiable
class TraceBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(link, route))

  val traceUnit: Instance[Trace] =
    Instantiate(new Trace(link, ball))

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
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new TraceBall(link, route)).blink, None)

}
