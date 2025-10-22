package framework.bbus.cmdrouter

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}

class CmdRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete)))
    val ballIdle = Input(Vec(numBalls, Bool()))
    val cmdReq_o = Decoupled(new BallRsIssue)
    val cmdResp_o = Vec(numBalls, Decoupled(new BallRsComplete))
  })

  val reqRouter = Module(new CmdReqRouter(numBalls))

  reqRouter.io.cmdReq_i <> io.cmdReq_i
  reqRouter.io.ballIdle := io.ballIdle
  io.cmdReq_o <> reqRouter.io.cmdReq_o

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }

  override lazy val desiredName = "CmdRouter"
}
