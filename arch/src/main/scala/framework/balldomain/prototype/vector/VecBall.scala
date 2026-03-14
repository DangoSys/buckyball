package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.prototype.vector.VecUnit
import framework.top.GlobalConfig

/**
 * VecBall
 */
@instantiable
class VecBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
    .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  val vecUnit: Instance[VecUnit] = Instantiate(new VecUnit(b))

  vecUnit.io.cmdReq <> io.cmdReq
  vecUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    vecUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    vecUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> vecUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}
