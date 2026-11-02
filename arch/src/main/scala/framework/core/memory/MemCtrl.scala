package framework.core.memory

import framework.core.memory.access.mmu.{PteRequest, PteResponse}
import framework.ip.memory.interlock.{Acknowledgement, CpuQuery, Maintenance}
import chisel3._
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import chisel3.util._

/** Core-owned translation and cache services. */
class MemCtrl(tracking: TrackingParam, walker: WalkerParam) extends Bundle {

  val cpuQuery    = Input(new CpuQuery(tracking))
  val cpuAllow    = Output(Bool())
  val maintenance = Decoupled(new Maintenance(tracking))
  val maintained  = Flipped(Decoupled(new Acknowledgement(tracking)))
  val pteRequest  = Decoupled(new PteRequest(walker))
  val pteResponse = Flipped(Decoupled(new PteResponse))
}
