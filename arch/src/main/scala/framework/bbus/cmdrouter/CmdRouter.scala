package framework.bbus.cmdrouter

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.bbus.BBusConfigIO

class CmdRouter(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete)))
    val ballIdle = Input(Vec(numBalls, Bool()))
    val cmdReq_o = Decoupled(new BallRsIssue)
    val cmdResp_o = Vec(numBalls, Decoupled(new BallRsComplete))
    val bbusConfig_o = Decoupled(new BBusConfigIO(numBalls))
  })

  val reqRouter = Module(new CmdReqRouter(numBalls))

  reqRouter.io.cmdReq_i <> io.cmdReq_i
  reqRouter.io.ballIdle := io.ballIdle
  io.cmdReq_o <> reqRouter.io.cmdReq_o

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }
  io.bbusConfig_o.valid := false.B
  io.bbusConfig_o.bits.src_bid := 0.U
  io.bbusConfig_o.bits.dst_bid := 0.U
  io.bbusConfig_o.bits.set := false.B
  when(io.cmdReq_i(b.emptyBallid).valid){
    io.bbusConfig_o.valid := true.B
    io.bbusConfig_o.bits.src_bid := io.cmdReq_i(b.emptyBallid).bits.cmd.special(5,0)
    io.bbusConfig_o.bits.dst_bid := io.cmdReq_i(b.emptyBallid).bits.cmd.special(11,6)
    io.bbusConfig_o.bits.set := io.cmdReq_i(b.emptyBallid).bits.cmd.special(12,12)
  }
  override lazy val desiredName = "CmdRouter"
}
