package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.bbus.cmdrouter.CmdReqRouter
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class CmdRouter(val b: GlobalConfig) extends Module {
  val numBalls = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val cmdReq_i  = Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b))))
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(b))))
    val cmdReq_o  = Decoupled(new BallRsIssue(b))
    val cmdResp_o = Vec(numBalls, Decoupled(new BallRsComplete(b)))

    val ballIdle = Input(Vec(numBalls, Bool()))
  })

  val reqRouter = Module(new CmdReqRouter(b, numBalls))

  reqRouter.io.cmdReq_i <> io.cmdReq_i
  reqRouter.io.ballIdle := io.ballIdle
  io.cmdReq_o <> reqRouter.io.cmdReq_o

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }
}
