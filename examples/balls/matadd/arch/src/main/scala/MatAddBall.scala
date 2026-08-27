package examples.balls.matadd

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig

@instantiable
class MatAddBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "MatAddBall")
    .getOrElse(throw new IllegalArgumentException("MatAddBall not found in config"))

  @public
  val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))

  def blink: BlinkIO = io
  def status = io.status
  dontTouch(io)

  val unit: Instance[MatAdd] = Instantiate(new MatAdd(b))
  unit.io.cmdReq <> io.cmdReq
  unit.io.cmdResp <> io.cmdResp
  unit.io.channelReady := io.channelReady
  unit.io.bankRead(0) <> io.bankRead(0)
  unit.io.bankRead(1) <> io.bankRead(1)
  unit.io.bankWrite(0) <> io.bankWrite(0)
  io.status <> unit.io.status
  io.subRobReq.valid   := false.B
  io.subRobReq.bits    := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
}
