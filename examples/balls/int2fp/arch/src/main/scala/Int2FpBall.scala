package examples.balls.int2fp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.balldomain.blink.mmio.MmioWrite
import framework.top.GlobalConfig

@instantiable
class Int2FpBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int2FpBall")
    .getOrElse(
      throw new IllegalArgumentException("Int2FpBall not found in config")
    )

  val inBW        = ballCommonConfig.inBW
  val outBW       = ballCommonConfig.outBW
  val mmioReadBW  = ballCommonConfig.mmioReadBW
  val mmioWriteBW = ballCommonConfig.mmioWriteBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW, mmioReadBW, mmioWriteBW))

  def blink: BlinkIO = io
  dontTouch(io)

  val int2fpUnit: Instance[Int2Fp] = Instantiate(new Int2Fp(b))

  int2fpUnit.io.cmdReq <> io.cmdReq
  int2fpUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    int2fpUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    int2fpUnit.io.bankWrite(i) <> io.bankWrite(i)
  }
  for (i <- 0 until mmioReadBW) {
    int2fpUnit.io.mmioRead(i) <> io.mmioRead(i)
  }

  io.status <> int2fpUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)

  MmioWrite.tieOff(io.mmioWrite)
}
