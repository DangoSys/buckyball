package examples.toy.balldomain.vecball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.vector.VecUnit

/**
 * VecBall - 遵守Blink协议的向量计算Ball
 */
class VecBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // 实例化VecUnit
  val vecUnit = Module(new VecUnit)

  // 连接命令接口
  vecUnit.io.cmdReq <> io.cmdReq
  vecUnit.io.cmdResp <> io.cmdResp

  // 连接SRAM读接口 - VecUnit需要从scratchpad读取数据
  for (i <- 0 until b.sp_banks) {
    vecUnit.io.sramRead(i) <> io.sramRead(i)
  }

  // 处理SRAM写接口 - VecUnit不写入scratchpad，所以tie off
  for (i <- 0 until b.sp_banks) {
    // 对于Flipped(SramWriteIO)，我们需要驱动req.valid和req.bits（输出）
    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits := DontCare
  }

  // 处理Accumulator读接口 - VecUnit不读accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramReadIO)，我们需要驱动req.valid, req.bits（输出）和resp.ready（输出）
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B
  }

  // 连接Accumulator写接口 - VecUnit向accumulator写入结果
  for (i <- 0 until b.acc_banks) {
    vecUnit.io.accWrite(i) <> io.accWrite(i)
  }

  // 连接Status信号 - 直接从内部单元获取
  io.status <> vecUnit.io.status

  override lazy val desiredName = "VecBall"
}
