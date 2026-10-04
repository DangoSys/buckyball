package framework.memdomain.verification

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import framework.top.GlobalConfig
import framework.system.core.accelerator.{Admission, BuckyballAccelerator}
import framework.system.core.rocket.RoCCIO
import framework.memdomain.backend.shared.{SharedMemBackend, SharedMemLayout}
import hier.core.rocket.AdmissionPorts
import hier.tile.TaskController
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.preflight.Params
import memcore.memory.cpu.PhysicalRegion
import memcore.bus.axi4

/** Actual NPU/ROB/task/translation/AXI join. Core and CMO peer use the declared BFM contract. */
@instantiable
class AdmissionSystem(
  b:                      GlobalConfig,
  prepared:               Params,
  coreIndex:              Int,
  workerSignatures:       Seq[BigInt],
  regions:                Seq[PhysicalRegion]
)(
  implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  private val tracking  = TrackingParams(addressBits = prepared.bus.addressBits)
  private val axiParams = axi4.Params(prepared.bus.addressBits, 128, 4)

  @public
  val io = IO(new Bundle {
    val core            = Flipped(new AdmissionPorts(tracking, nPMPs, prepared.bus))
    val controller      = new RoCCIO(64)
    val controllerSatp  = Input(UInt(64.W))
    val axi             = new memcore.bus.axi4.Port(axiParams)
    val npuIdle         = Output(Bool())
    val halted          = Output(Bool())
    val fault           = Output(new framework.memdomain.frontend.mem.dma.DmaStatus)
    val faultTag        = Output(UInt(tracking.idBits.W))
    val workDrained     = Output(Bool())
    val blockedResponse = Input(Bool())
  })

  val admission   = Instantiate(new Admission(b, tracking, prepared, regions, axiParams))
  val accelerator = Instantiate(new BuckyballAccelerator(b))
  val tasks       = Instantiate(new TaskController((1 to workerSignatures.size), workerSignatures, b.memDomain.nCores))
  admission.io.core <> io.core
  accelerator.io.cmd <> admission.io.npuCommand
  admission.io.npuResponse <> accelerator.io.resp
  admission.io.allocation.valid        := accelerator.io.allocation.valid
  admission.io.allocation.bits         := accelerator.io.allocation.bits.rob_id
  admission.io.retired                 := accelerator.io.retired
  admission.io.npuFault                := accelerator.io.fault
  admission.io.npuBusy                 := !accelerator.io.idle
  admission.io.npuInterrupt            := accelerator.io.interrupt
  admission.io.footprints              := accelerator.io.footprints
  admission.io.dma <> accelerator.io.dma
  accelerator.io.hartid                := coreIndex.U
  accelerator.io.sharedBankOwnerHartId := 0.U
  accelerator.io.barrier_release       := accelerator.io.barrier_arrive
  admission.io.task <> tasks.io.ports(coreIndex)
  tasks.io.ports(0) <> io.controller
  tasks.io.satp                        := io.controllerSatp
  for (i <- 1 until b.memDomain.nCores if i != coreIndex) {
    tasks.io.ports(i).cmd.valid  := false.B
    tasks.io.ports(i).cmd.bits   := 0.U.asTypeOf(tasks.io.ports(i).cmd.bits)
    tasks.io.ports(i).resp.ready := true.B
    tasks.io.ports(i).exception  := false.B
  }
  io.npuIdle := accelerator.io.idle
  io.halted   := admission.io.halted; io.fault         := admission.io.fault
  io.faultTag := admission.io.faultTag; io.workDrained := admission.io.workDrained

  val shared   = Instantiate(new SharedMemBackend(b, useMesh = true))
  shared.io.config <> accelerator.io.shared_config
  shared.io.mvover <> accelerator.io.mvover
  val channels = SharedMemLayout.channelPerHart(b)
  val endpoint = b.memDomain.computeCoreIds.indexOf(coreIndex)
  require(endpoint >= 0)
  for (i <- shared.io.mem_req.indices) {
    if (i >= endpoint * channels && i < (endpoint + 1) * channels) {
      shared.io.mem_req(i) <> accelerator.io.shared_mem_req(i - endpoint * channels)
    } else {
      val q = shared.io.mem_req(i)
      q.read.req.valid  := false.B; q.read.req.bits  := 0.U.asTypeOf(q.read.req.bits); q.read.resp.ready   := true.B
      q.write.req.valid := false.B; q.write.req.bits := 0.U.asTypeOf(q.write.req.bits); q.write.resp.ready := true.B
      q.bank_id         := 0.U; q.group_id           := 0.U; q.is_shared                                   := false.B; q.hart_id := 0.U; q.rob_id := 0.U; q.inst_id := 0.U
    }
  }
  for (i <- 0 until b.memDomain.nCores) {
    shared.io.query_valid(i)    := (if (i == coreIndex) accelerator.io.shared_query_valid else false.B)
    shared.io.query_hart_id(i)  := i.U
    shared.io.query_vbank_id(i) := (if (i == coreIndex) accelerator.io.shared_query_vbank_id else 0.U)
    if (i == coreIndex) shared.io.localBanks(i) <> accelerator.io.meshLocalBank
    else {
      shared.io.localBanks(i).request.ready  := false.B
      shared.io.localBanks(i).response.valid := false.B
      shared.io.localBanks(i).response.bits  := 0.U.asTypeOf(shared.io.localBanks(i).response.bits)
    }
  }
  accelerator.io.shared_query_group_count := shared.io.query_group_count(coreIndex)
  accelerator.io.shared_bank_hashes.foreach(_ := shared.io.bank_hashes.get)

  io.axi.aw <> admission.io.axi.aw
  io.axi.w <> admission.io.axi.w
  io.axi.ar <> admission.io.axi.ar
  admission.io.axi.r.valid := io.axi.r.valid && !io.blockedResponse
  admission.io.axi.r.bits  := io.axi.r.bits
  io.axi.r.ready           := admission.io.axi.r.ready && !io.blockedResponse
  admission.io.axi.b.valid := io.axi.b.valid && !io.blockedResponse
  admission.io.axi.b.bits  := io.axi.b.bits
  io.axi.b.ready           := admission.io.axi.b.ready && !io.blockedResponse
}
