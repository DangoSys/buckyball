package examples.balls.mxfp2int

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.blink.{BlinkIO, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig

/**
 * Mxfp2IntBall - MXFP4 to INT8 dequantizer.
 *
 * Wraps PipelinedMxfp2Int and connects to the Blink protocol,
 * including the MMIO read channel for per-block E8M0 scales.
 */
@instantiable
class Mxfp2IntBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "Mxfp2IntBall")
    .getOrElse(throw new IllegalArgumentException("Mxfp2IntBall not found in config"))

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io
  dontTouch(io)

  val mxfp2intUnit: Instance[PipelinedMxfp2Int] = Instantiate(new PipelinedMxfp2Int(b))

  mxfp2intUnit.io.cmdReq <> io.cmdReq
  mxfp2intUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    mxfp2intUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    mxfp2intUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> mxfp2intUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)

  io.mmioRead <> mxfp2intUnit.io.mmioRead
}
