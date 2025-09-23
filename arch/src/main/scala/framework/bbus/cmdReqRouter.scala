package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}

/**
 * 命令路由器 - 解析命令并输出目标Ball ID和路由后的命令
 * 优化版本：只对空闲的ball进行仲裁
 *
 * 输入：
 * - cmdReq：上游命令输入
 * - ballIdle：各ball的空闲状态信号
 *
 * 输出：
 * - ballId：目标Ball ID
 */
class CmdReqRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val ballIdle = Input(Vec(numBalls, Bool()))  // 各ball的空闲状态

    val cmdReq_o = Decoupled(new BallRsIssue)
  })

  val arbiter = Module(new RRArbiter(new BallRsIssue, numBalls))

  // 只有当ball空闲时才参与仲裁
  for (i <- 0 until numBalls) {
    arbiter.io.in(i).valid := io.cmdReq_i(i).valid && io.ballIdle(i)
    arbiter.io.in(i).bits := io.cmdReq_i(i).bits
    io.cmdReq_i(i).ready := arbiter.io.in(i).ready && io.ballIdle(i)
  }

  io.cmdReq_o <> arbiter.io.out

  override lazy val desiredName = "CmdReqRouter"
}
