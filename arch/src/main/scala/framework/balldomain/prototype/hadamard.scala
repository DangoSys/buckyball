package framework.balldomain.prototype.hadamard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.prototype.hadamard.HadamardUnit
import framework.top.GlobalConfig

@instantiable
class HadamardBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "HadamardBall")
    .getOrElse(throw new IllegalArgumentException("HadamardBall not found in config"))
  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  val hadamardUnit: Instance[HadamardUnit] = Instantiate(new HadamardUnit(b))

  hadamardUnit.io.cmdReq <> io.cmdReq
  hadamardUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    hadamardUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    hadamardUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> hadamardUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}