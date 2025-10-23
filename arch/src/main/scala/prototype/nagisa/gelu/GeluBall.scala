package prototype.nagisa.gelu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.nagisa.gelu.GeluUnit

/**
 * GeluBall - 遵守Blink协议的GELU计算Ball
 */
class GeluBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // 实例化GeluUnit
  val geluUnit = Module(new GeluUnit)

  // 连接命令接口
  geluUnit.io.cmdReq <> io.cmdReq
  geluUnit.io.cmdResp <> io.cmdResp

  // 连接SRAM读写接口
  for (i <- 0 until b.sp_banks) {
    geluUnit.io.sramRead(i) <> io.sramRead(i)
    geluUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // 连接Accumulator读写接口
  for (i <- 0 until b.acc_banks) {
    geluUnit.io.accRead(i) <> io.accRead(i)
    geluUnit.io.accWrite(i) <> io.accWrite(i)
  }

  // 连接Status信号
  io.status <> geluUnit.io.status

  override lazy val desiredName = "GeluBall"
}
