package framework.balldomain.prototype.quant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.top.GlobalConfig

@instantiable
class QuantBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "QuantBall")
    .getOrElse(throw new IllegalArgumentException("QuantBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val quantUnit: Instance[Quant] = Instantiate(new Quant(b))

  quantUnit.io.cmdReq <> io.cmdReq
  quantUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    quantUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    quantUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> quantUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}
