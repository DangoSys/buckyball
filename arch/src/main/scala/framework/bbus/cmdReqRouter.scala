package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}

/**
 * 命令路由器 - 解析命令并输出目标Ball ID和路由后的命令
 *
 * 输入：
 * - cmdReq：上游命令输入
 *
 * 输出：
 * - ballId：目标Ball ID
 */
class CmdReqRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Flipped(Decoupled(new BallRsIssue))

    val ballId = Output(UInt(log2Ceil(numBalls).W))
    val cmdReq_o = Decoupled(new BallRsIssue)
  })

  io.ballId := io.cmdReq_i.bits.cmd.bid

  io.cmdReq_o.valid := io.cmdReq_i.valid && (io.ballId < numBalls.U)
  io.cmdReq_o.bits := io.cmdReq_i.bits

  io.cmdReq_i.ready := io.cmdReq_o.ready

  override lazy val desiredName = "CmdReqRouter"
}
