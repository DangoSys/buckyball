package framework.memdomain.pmc

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.rs.{MemRsIssue, MemRsComplete}

class MemCyclePMC(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ldReq_i = Input(Valid(new MemRsIssue))
    val stReq_i = Input(Valid(new MemRsIssue))
    val ldResp_o = Input(Valid(new MemRsComplete))
    val stResp_o = Input(Valid(new MemRsComplete))
    val ldTotalCycles = Output(UInt(64.W))
    val stTotalCycles = Output(UInt(64.W))
  })

  val cycleCounter = RegInit(0.U(64.W))
  cycleCounter := cycleCounter + 1.U

  val startTime = Reg(Vec(b.rob_entries, UInt(64.W)))
  val ldTotalCycles = RegInit(0.U(64.W))
  val stTotalCycles = RegInit(0.U(64.W))

  when(io.ldReq_i.valid) {
    startTime(io.ldReq_i.bits.rob_id) := cycleCounter
  }

  when(io.stReq_i.valid) {
    startTime(io.stReq_i.bits.rob_id) := cycleCounter
  }

  when(io.ldResp_o.valid) {
    val robId = io.ldResp_o.bits.rob_id
    val elapsed = cycleCounter - startTime(robId)
    ldTotalCycles := ldTotalCycles + elapsed
    printf("[PMC] Load completed, elapsed: %d cycles\n", elapsed)
  }

  when(io.stResp_o.valid) {
    val robId = io.stResp_o.bits.rob_id
    val elapsed = cycleCounter - startTime(robId)
    stTotalCycles := stTotalCycles + elapsed
    printf("[PMC] Store completed, elapsed: %d cycles\n", elapsed)
  }

  io.ldTotalCycles := ldTotalCycles
  io.stTotalCycles := stTotalCycles
}
