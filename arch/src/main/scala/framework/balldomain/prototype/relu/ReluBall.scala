package framework.balldomain.prototype.relu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.prototype.relu.PipelinedRelu
import framework.top.GlobalConfig

/**
 * ReluBall - A ReLU computation Ball that complies with the blink protocol.
 * Behavior: Read data from Scratchpad, perform element-wise ReLU (set negative values to 0),
 * then write back to Scratchpad.
 */
@instantiable
class ReluBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "ReluBall")
    .getOrElse(throw new IllegalArgumentException("ReluBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val reluUnit: Instance[PipelinedRelu] = Instantiate(new PipelinedRelu(b))

  reluUnit.io.cmdReq <> io.cmdReq
  reluUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    reluUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    reluUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> reluUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}
