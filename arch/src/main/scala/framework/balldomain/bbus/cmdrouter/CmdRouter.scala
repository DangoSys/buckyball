package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
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

  val arbiter = Module(new RRArbiter(new BallRsIssue(b), numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i).valid := io.cmdReq_i(i).valid && io.ballIdle(i)
    arbiter.io.in(i).bits  := io.cmdReq_i(i).bits
    io.cmdReq_i(i).ready   := arbiter.io.in(i).ready && io.ballIdle(i)
  }

  io.cmdReq_o <> arbiter.io.out

  for (i <- 0 until numBalls) {
    io.cmdResp_o(i) <> io.cmdResp_i(i)
  }
}
