package examples.toy.balldomain.transposeball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.transpose.PipelinedTransposer

/**
 * TransposeBall - 遵守Blink协议的转置计算Ball
 */
class TransposeBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // 实例化PipelinedTransposer
  val transposeUnit = Module(new PipelinedTransposer)

  // 连接命令接口
  transposeUnit.io.cmdReq <> io.cmdReq
  transposeUnit.io.cmdResp <> io.cmdResp

  // 连接SRAM读接口 - Transpose需要从scratchpad读取数据
  for (i <- 0 until b.sp_banks) {
    transposeUnit.io.sramRead(i) <> io.sramRead(i)
  }

  // 连接SRAM写接口 - Transpose需要写入scratchpad
  for (i <- 0 until b.sp_banks) {
    transposeUnit.io.sramWrite(i) <> io.sramWrite(i)
  }

  // 处理Accumulator读接口 - Transpose不读accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramReadIO)，我们需要驱动req.valid, req.bits（输出）和resp.ready（输出）
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B
  }

  // 处理Accumulator写接口 - Transpose不写accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramWriteIO)，我们需要驱动req.valid和req.bits（输出）
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits := DontCare
  }

  // 连接Status信号 - 直接从内部单元获取
  io.status <> transposeUnit.io.status

  override lazy val desiredName = "TransposeBall"
}
