package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.BallRsComplete

class CmdRespRouter(b: GlobalConfig, numBalls: Int) extends Module {

  val io = IO(new Bundle {
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(b))))
    val cmdResp_o = Decoupled(new BallRsComplete(b))
  })

  val arbiter = Module(new RRArbiter(new BallRsComplete(b), numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i) <> io.cmdResp_i(i)
  }

  io.cmdResp_o <> arbiter.io.out

}
