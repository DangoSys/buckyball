package framework.balldomain.prototype.int2fp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig

@instantiable
class Int2FpBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "Int2FpBall")
    .getOrElse(throw new IllegalArgumentException("Int2FpBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val int2fpUnit: Instance[Int2Fp] = Instantiate(new Int2Fp(b))

  int2fpUnit.io.cmdReq <> io.cmdReq
  int2fpUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    int2fpUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    int2fpUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> int2fpUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)

  // MMIO: this Ball does not consume MMIO metadata
  MmioRead.tieOff(io.mmioRead)
}
