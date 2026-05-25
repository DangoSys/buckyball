package framework.balldomain.prototype.mxfp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.blink.{BlinkIO, HasBlink, SubRobRow}
import framework.top.GlobalConfig

/**
 * MxfpBall - MXFP conversion ball wrapper.
 *
 * This module only wraps the inner PipelinedMxfp execution unit
 * and connects it to the blink protocol.
 */
@instantiable
class MxfpBall(val b: GlobalConfig) extends Module with HasBlink {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "MxfpBall")
    .getOrElse(throw new IllegalArgumentException("MxfpBall not found in config"))

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink: BlinkIO = io

  val mxfpUnit: Instance[PipelinedMxfp] = Instantiate(new PipelinedMxfp(b))

  mxfpUnit.io.cmdReq <> io.cmdReq
  mxfpUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    mxfpUnit.io.bankRead(i) <> io.bankRead(i)
  }

  for (i <- 0 until outBW) {
    mxfpUnit.io.bankWrite(i) <> io.bankWrite(i)
  }

  io.status <> mxfpUnit.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits := SubRobRow.tieOff(b)
}
