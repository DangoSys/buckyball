package examples.balls.mxmm

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.top.GlobalConfig
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}

@instantiable
class MxmmBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {
  @public val io = IO(new BlinkIO(b, 2, 1))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  val unit = Instantiate(new Unit(b))
  unit.io.cmdReq <> io.cmdReq
  unit.io.cmdResp <> io.cmdResp
  unit.io.channelReady := io.channelReady
  unit.io.bankRead <> io.bankRead
  unit.io.bankWrite <> io.bankWrite
  io.status <> unit.io.status
  io.subRobReq.valid   := false.B
  io.subRobReq.bits    := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)
}
