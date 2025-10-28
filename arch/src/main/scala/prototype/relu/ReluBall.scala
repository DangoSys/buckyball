package prototype.relu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.relu.PipelinedRelu

/** ReluBall - 遵守 Blink 协议的 ReLU 计算 Ball 行为：从 Scratchpad 读取数据，逐元素做 ReLU（负数置
  * 0），再写回 Scratchpad。
  */
class ReluBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // 满足 BallRegist 的要求
  def Blink: Blink = io

  // 实例化 PipelinedRelu 计算单元
  private val reluUnit = Module(new PipelinedRelu[UInt])

  // 连接命令接口
  reluUnit.io.cmdReq <> io.cmdReq
  reluUnit.io.cmdResp <> io.cmdResp

  // 连接 Scratchpad SRAM 读写接口
  for (i <- 0 until b.sp_banks) {
    reluUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    reluUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator 读接口（ReLU 不访问 accumulator，tie-off）
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Accumulator 写接口（ReLU 不写 accumulator，tie-off）
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // 透传状态信号
  io.status <> reluUnit.io.status

  override lazy val desiredName: String = "ReluBall"
}
