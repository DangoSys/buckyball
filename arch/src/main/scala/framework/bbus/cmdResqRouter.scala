package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}

/**
 * 命令响应路由器 - 仲裁多个Ball的响应输出，并输出到BBus
 */
class CmdRespRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete)))

    val cmdResp_o = Decoupled(new BallRsComplete)
  })

  val arbiter = Module(new RRArbiter(new BallRsComplete, numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i) <> io.cmdResp_i(i)
  }

  io.cmdResp_o <> arbiter.io.out

  override lazy val desiredName = "CmdRespRouter"
}
