package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.nagisa.softmax.SoftmaxUnit

/**
 * SoftmaxBall - Softmax computation ball following Blink protocol
 */
class SoftmaxBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate SoftmaxUnit
  val softmaxUnit = Module(new SoftmaxUnit)

  // Connect command interface
  softmaxUnit.io.cmdReq <> io.cmdReq
  softmaxUnit.io.cmdResp <> io.cmdResp

  // Connect SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    softmaxUnit.io.sramRead(i) <> io.sramRead(i)
    softmaxUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // Connect Accumulator read/write interface
  for (i <- 0 until b.acc_banks) {
    softmaxUnit.io.accRead(i) <> io.accRead(i)
    softmaxUnit.io.accWrite(i) <> io.accWrite(i)
  }

  // Connect Status signals
  io.status <> softmaxUnit.io.status

  override lazy val desiredName = "SoftmaxBall"
}
