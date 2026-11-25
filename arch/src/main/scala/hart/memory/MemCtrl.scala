package framework.hart.memory

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}

import framework.hart.memory.access.mmu.{PteRequest, PteResponse}
import framework.generic.memory.interlock.{Acknowledgement, CpuQuery, Maintenance}
import chisel3._
import framework.generic.memory.interlock.{Params => TrackingParam}
import chisel3.util._

/** Hart-owned translation and cache services. */
class MemCtrl(tracking: TrackingParam, walker: WalkerParam) extends Bundle {

  val cpuQuery     = Input(new CpuQuery(tracking))
  val cpuAllow     = Output(Bool())
  val maintenance  = Decoupled(new Maintenance(tracking))
  val maintained   = Flipped(Decoupled(new Acknowledgement(tracking)))
  // The owner of DMA coherence asks about one line; the memory system answers in the same cycle.
  val lineAddress  = Output(UInt(tracking.addressBits.W))
  val lineWrite    = Output(Bool())
  val lineMaintain = Input(Bool())
  val drained      = Input(Bool())
  val pteRequest   = Decoupled(new PteRequest(walker))
  val pteResponse  = Flipped(Decoupled(new PteResponse))
}
