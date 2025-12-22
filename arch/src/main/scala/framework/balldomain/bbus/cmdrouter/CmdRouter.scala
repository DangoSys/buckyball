package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.bbus.BBusConfigIO

class CmdRouter(val parameter: BallDomainParam, val numBalls: Int) extends Module {

  val io = IO(new Bundle {
    val cmdReq_i     = Vec(numBalls, Flipped(Decoupled(new BallRsIssue(parameter))))
    val cmdResp_i    = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(parameter))))
    val ballIdle     = Input(Vec(numBalls, Bool()))
    val cmdReq_o     = Decoupled(new BallRsIssue(parameter))
    val cmdResp_o    = Vec(numBalls, Decoupled(new BallRsComplete(parameter)))
    val bbusConfig_o = Decoupled(new BBusConfigIO(numBalls))
  })

  val reqRouter = Module(new CmdReqRouter(parameter, numBalls))

  reqRouter.io.cmdReq_i <> io.cmdReq_i
  reqRouter.io.ballIdle := io.ballIdle
  io.cmdReq_o <> reqRouter.io.cmdReq_o

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }
  io.bbusConfig_o.valid        := false.B
  io.bbusConfig_o.bits.src_bid := 0.U
  io.bbusConfig_o.bits.dst_bid := 0.U
  io.bbusConfig_o.bits.set     := false.B
  when(io.cmdReq_i(parameter.emptyBallid).valid) {
    io.bbusConfig_o.valid        := true.B
    io.bbusConfig_o.bits.src_bid := io.cmdReq_i(parameter.emptyBallid).bits.cmd.special(5, 0)
    io.bbusConfig_o.bits.dst_bid := io.cmdReq_i(parameter.emptyBallid).bits.cmd.special(11, 6)
    io.bbusConfig_o.bits.set     := io.cmdReq_i(parameter.emptyBallid).bits.cmd.special(12, 12)
  }
  override lazy val desiredName = "CmdRouter"
}
