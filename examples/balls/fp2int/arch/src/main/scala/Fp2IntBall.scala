package examples.balls.fp2int

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig

@instantiable
class Fp2IntBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "Fp2IntBall")
    .getOrElse(throw new IllegalArgumentException("Fp2IntBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io
  dontTouch(io)

  val fp2intUnit: Instance[Fp2Int] = Instantiate(new Fp2Int(b))

  fp2intUnit.io.cmdReq <> io.cmdReq
  fp2intUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    fp2intUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    fp2intUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> fp2intUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)

  MmioRead.tieOff(io.mmioRead)
}
