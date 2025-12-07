package framework.balldomain.bbus.cmdrouter

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.rs.{BallRsIssue, BallRsComplete}

class CmdReqRouter(numBalls: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val ballIdle = Input(Vec(numBalls, Bool()))
    val cmdReq_o = Decoupled(new BallRsIssue)
  })

  val arbiter = Module(new RRArbiter(new BallRsIssue, numBalls))

  for (i <- 0 until numBalls) {
    arbiter.io.in(i).valid := io.cmdReq_i(i).valid && io.ballIdle(i)
    arbiter.io.in(i).bits := io.cmdReq_i(i).bits
    io.cmdReq_i(i).ready := arbiter.io.in(i).ready && io.ballIdle(i)
  }

  io.cmdReq_o <> arbiter.io.out

  override lazy val desiredName = "CmdReqRouter"
}
