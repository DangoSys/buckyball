package examples.toy.balldomain.matrixball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.matrix.BBFP_Control

/**
 * MatrixBall - 遵守Blink协议的矩阵计算Ball
 */
class MatrixBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // 实例化BBFP_Control
  val matrixUnit = Module(new BBFP_Control)

  // 连接命令接口
  matrixUnit.io.cmdReq <> io.cmdReq
  matrixUnit.io.cmdResp <> io.cmdResp

  // 设置is_matmul_ws信号
  matrixUnit.io.is_matmul_ws := false.B  // TODO:

  // 连接SRAM读接口 - MatrixBall需要从scratchpad读取数据
  for (i <- 0 until b.sp_banks) {
    matrixUnit.io.sramRead(i) <> io.sramRead(i)
  }

  // 连接SRAM写接口 - MatrixBall需要写入scratchpad
  for (i <- 0 until b.sp_banks) {
    matrixUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // 处理Accumulator读接口 - MatrixBall不读accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramReadIO)，我们需要驱动req.valid, req.bits（输出）和resp.ready（输出）
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B
  }

  // 连接Accumulator写接口 - MatrixBall向accumulator写入结果
  for (i <- 0 until b.acc_banks) {
    matrixUnit.io.accWrite(i) <> io.accWrite(i)
  }

  override lazy val desiredName = "MatrixBall"
}
