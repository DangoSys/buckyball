package examples.toy.balldomain.reluball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.relu.ReluAccelerator

/** ReluBall - 遵守Blink协议的ReLU计算Ball
  */
class ReluBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // 实例化ReluAccelerator
  val reluUnit = Module(new ReluAccelerator)

  // 连接命令接口
  reluUnit.io.cmdReq <> io.cmdReq
  reluUnit.io.cmdResp <> io.cmdResp

  // 连接SRAM读接口 - ReLU需要从scratchpad读取数据
  for (i <- 0 until b.sp_banks) {
    reluUnit.io.sramRead(i) <> io.sramRead(i)
  }

  // 连接SRAM写接口 - ReLU需要写入scratchpad
  for (i <- 0 until b.sp_banks) {
    reluUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // 处理Accumulator读接口 - ReLU不读accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B
  }

  // 处理Accumulator写接口 - ReLU不写accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits := DontCare
  }

  override lazy val desiredName = "ReluBall"
}
