package examples.toy.balldomain.layernormball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.nagisa.layernorm.LayerNormUnit

/**
 * LayerNormBall - LayerNorm computation ball following Blink protocol
 */
class LayerNormBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate LayerNormUnit
  val layerNormUnit = Module(new LayerNormUnit)

  // Connect command interface
  layerNormUnit.io.cmdReq <> io.cmdReq
  layerNormUnit.io.cmdResp <> io.cmdResp

  // Connect SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    layerNormUnit.io.sramRead(i) <> io.sramRead(i)
    layerNormUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // Connect Accumulator read/write interface
  for (i <- 0 until b.acc_banks) {
    layerNormUnit.io.accRead(i) <> io.accRead(i)
    layerNormUnit.io.accWrite(i) <> io.accWrite(i)
  }

  // Connect Status signals
  io.status <> layerNormUnit.io.status

  override lazy val desiredName = "LayerNormBall"
}
