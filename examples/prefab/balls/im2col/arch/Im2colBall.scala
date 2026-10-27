package examples.balls.im2col

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.im2col.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.im2col.configs.Im2colBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.{BlinkIO, HasBlink}
import examples.balls.im2col.Im2col

/**
 * Im2colBall - An Im2col computation Ball that complies with the Blink protocol
 */
@instantiable
class Im2colBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = Im2colBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  val ballCommonConfig = ball

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  // Instantiate Im2col
  val im2colUnit: Instance[Im2col] =
    Instantiate(new Im2col(fp, mp, bp, ball, ballParam))

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
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Im2colBall(fp, mp, bp, id)).blink, None)

}
