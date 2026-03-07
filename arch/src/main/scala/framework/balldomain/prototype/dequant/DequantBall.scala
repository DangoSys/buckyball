package framework.balldomain.prototype.dequant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink}
import framework.top.GlobalConfig

@instantiable
class DequantBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "DequantBall")
    .getOrElse(throw new IllegalArgumentException("DequantBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val dequantUnit: Instance[Dequant] = Instantiate(new Dequant(b))

  dequantUnit.io.cmdReq <> io.cmdReq
  dequantUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    dequantUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    dequantUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> dequantUnit.io.status
}
