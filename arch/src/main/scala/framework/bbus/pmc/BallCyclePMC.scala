package framework.bbus.pmc

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.rs.{BallRsIssue, BallRsComplete}

class BallCyclePMC(numBalls: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq_i = Input(Vec(numBalls, Valid(new BallRsIssue)))
    val cmdResp_o = Input(Vec(numBalls, Valid(new BallRsComplete)))
    val totalCycles = Output(Vec(numBalls, UInt(64.W)))
  })

  val cycleCounter = RegInit(0.U(64.W))
  cycleCounter := cycleCounter + 1.U

  val startTime = Reg(Vec(b.rob_entries, UInt(64.W)))
  val ballTotalCycles = RegInit(VecInit(Seq.fill(numBalls)(0.U(64.W))))

  for (i <- 0 until numBalls) {
    when(io.cmdReq_i(i).valid) {
      startTime(io.cmdReq_i(i).bits.rob_id) := cycleCounter
    }
  }

  for (i <- 0 until numBalls) {
    when(io.cmdResp_o(i).valid) {
      val robId = io.cmdResp_o(i).bits.rob_id
      val elapsed = cycleCounter - startTime(robId)
      ballTotalCycles(i) := ballTotalCycles(i) + elapsed
      printf("[PMC] Ball %d completed task, elapsed: %d cycles\n", i.U, elapsed)
    }
  }

  io.totalCycles := ballTotalCycles
}
