package framework.balldomain.bbus.pmc

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.BallRsIssue
import framework.balldomain.rs.BallRsComplete
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class BallCyclePMC(val b: GlobalConfig) extends Module {
  val numBalls = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val cmdReq_i    = Input(Vec(numBalls, Valid(new BallRsIssue(b))))
    val cmdResp_o   = Input(Vec(numBalls, Valid(new BallRsComplete(b))))
    val totalCycles = Output(Vec(numBalls, UInt(64.W)))
  })

  val cycleCounter = RegInit(0.U(64.W))
  cycleCounter := cycleCounter + 1.U

  val startTime       = Reg(Vec(b.frontend.rob_entries, UInt(64.W)))
  val ballTotalCycles = RegInit(VecInit(Seq.fill(numBalls)(0.U(64.W))))

  // Per-Ball DPI-C trace modules
  val pmcTraces = Seq.fill(numBalls)(Module(new PMCTraceDPI))
  for (pt <- pmcTraces) {
    pt.io.ball_id := 0.U
    pt.io.rob_id  := 0.U
    pt.io.elapsed := 0.U
    pt.io.enable  := false.B
  }

  for (i <- 0 until numBalls) {
    when(io.cmdReq_i(i).valid) {
      startTime(io.cmdReq_i(i).bits.rob_id) := cycleCounter
    }
  }

  for (i <- 0 until numBalls) {
    when(io.cmdResp_o(i).valid) {
      val robId   = io.cmdResp_o(i).bits.rob_id
      val elapsed = cycleCounter - startTime(robId)
      ballTotalCycles(i) := ballTotalCycles(i) + elapsed

      // DPI-C trace output
      pmcTraces(i).io.ball_id := i.U
      pmcTraces(i).io.rob_id  := robId
      pmcTraces(i).io.elapsed := elapsed
      pmcTraces(i).io.enable  := true.B
    }
  }

  io.totalCycles := ballTotalCycles
}
