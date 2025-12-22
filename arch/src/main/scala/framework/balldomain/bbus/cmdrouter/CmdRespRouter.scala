package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.rs.BallRsComplete

class CmdRespRouter(val parameter: BallDomainParam, val numBalls: Int) extends Module {

  val io = IO(new Bundle {
    val cmdResp_i = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(parameter))))
    val cmdResp_o = Decoupled(new BallRsComplete(parameter))
  })

  val arbiter = Module(new RRArbiter(new BallRsComplete(parameter), numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i) <> io.cmdResp_i(i)
  }

  io.cmdResp_o <> arbiter.io.out

  override lazy val desiredName = "CmdRespRouter"
}
