package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  val systolicArrayUnit: Instance[SystolicArrayUnit] = Instantiate(new SystolicArrayUnit(b))

  systolicArrayUnit.io.cmdReq <> io.cmdReq
  systolicArrayUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    systolicArrayUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    systolicArrayUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> systolicArrayUnit.io.status
}
