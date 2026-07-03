package examples.balls.transpose

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import examples.balls.transpose.Transpose
import framework.top.GlobalConfig

@instantiable
class TransposeBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "TransposeBall")
    .getOrElse(throw new IllegalArgumentException("TransposeBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val transposeUnit: Instance[Transpose] = Instantiate(new Transpose(b))

  transposeUnit.io.cmdReq <> io.cmdReq
  transposeUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    transposeUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    transposeUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> transposeUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)

  // MMIO: this Ball does not consume MMIO metadata
  MmioRead.tieOff(io.mmioRead)
}
