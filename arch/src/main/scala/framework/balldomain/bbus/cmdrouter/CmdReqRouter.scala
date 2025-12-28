package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.BallRsIssue

class CmdReqRouter(val b: GlobalConfig, val numBalls: Int) extends Module {

  val io = IO(new Bundle {
    val cmdReq_i = Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b))))
    val ballIdle = Input(Vec(numBalls, Bool()))
    val cmdReq_o = Decoupled(new BallRsIssue(b))
  })

  val arbiter = Module(new RRArbiter(new BallRsIssue(b), numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i).valid := io.cmdReq_i(i).valid && io.ballIdle(i)
    arbiter.io.in(i).bits  := io.cmdReq_i(i).bits
    io.cmdReq_i(i).ready   := arbiter.io.in(i).ready && io.ballIdle(i)
  }

  io.cmdReq_o <> arbiter.io.out

}
