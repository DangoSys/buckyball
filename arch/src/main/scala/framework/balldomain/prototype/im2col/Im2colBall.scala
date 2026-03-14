package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.prototype.im2col.Im2col
import framework.top.GlobalConfig

/**
 * Im2colBall - An Im2col computation Ball that complies with the Blink protocol
 */
class Im2colBall(val b: GlobalConfig) extends Module with HasBlink {
  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  // Instantiate Im2col
  val im2colUnit: Instance[Im2col] = Instantiate(new Im2col(b))

  // Connect command interface
  im2colUnit.io.cmdReq <> io.cmdReq
  im2colUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    im2colUnit.io.bankRead(i) <> io.bankRead(i)
  }

  // Connect SRAM write interface - Im2col needs to write to scratchpad
  for (i <- 0 until outBW) {
    im2colUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> im2colUnit.io.status

  // Ball does not use SubROB: tie off subRobReq
  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}
