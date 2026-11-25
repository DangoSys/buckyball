package examples.balls.im2col

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.im2col.{Params => ParametersProto}

import examples.balls.im2col.configs.Im2colBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import examples.balls.im2col.Im2col

/**
 * Im2colBall - An Im2col computation Ball that complies with the Blink protocol
 */
@instantiable
class Im2colBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val ballParam = Im2colBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(link, route))

  // Instantiate Im2col
  val im2colUnit: Instance[Im2col] =
    Instantiate(new Im2col(link, ball, ballParam))

  // Connect command interface
  im2colUnit.io.cmdReq <> blink.cmdReq
  im2colUnit.io.cmdResp <> blink.cmdResp

  for (i <- 0 until inBW) {
    im2colUnit.io.bankRead(i) <> blink.bankRead(i)
  }

  // Connect SRAM write interface - Im2col needs to write to scratchpad
  for (i <- 0 until outBW) {
    im2colUnit.io.bankWrite(i) <> blink.bankWrite(i)
  }

  // Connect Status signals - directly obtained from internal unit
  blink.status <> im2colUnit.io.status

  // MMIO: this Ball does not consume MMIO metadata
}

object Im2colBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Im2colBall(link, route)).blink, None)

}
