package examples.balls.int2fp

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class Int2FpBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int2FpBall")
    .getOrElse(throw new IllegalArgumentException("Int2FpBall not found in config"))

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val unit: Instance[Int2Fp] = Instantiate(new Int2Fp(b))
  unit.io.cmdReq <> io.cmdReq
  unit.io.cmdResp <> io.cmdResp
  for (i <- 0 until mapping.inBW) unit.io.bankRead(i) <> io.bankRead(i)
  for (i <- 0 until mapping.outBW) unit.io.bankWrite(i) <> io.bankWrite(i)
  io.status <> unit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)
}
