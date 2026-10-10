package framework.system.core.clink

import chisel3._
import chisel3.util._
import framework.ant.{Params => AntParams, LocalPort}
import framework.top.GlobalConfig
import framework.frontend.globalrs.RobAllocation
import framework.memdomain.frontend.mem.Footprint
import framework.memdomain.frontend.mem.dma.DmaPort
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.backend.shared.SharedMemLayout
import framework.system.core.accelerator.AntMemoryPort
import framework.system.core.rocket.{CpuParams, RoCCCommandBB, RoCCResponseBB}
import hier.core.rocket.{AdmissionPorts, CommandSnapshot, RobFault}
import memcore.bus.chi
import memcore.memory.interlock.{Params => TrackingParams}

/** Task address-space binding and identity; accepted work drains before context reuse. */
class Control(p: AntParams, tracking: TrackingParams, pmps: Int)(implicit val cpuParams: CpuParams) extends Bundle {
  val hartId      = Input(UInt(64.W))
  val shmOwner    = Input(UInt(64.W))
  val fingerprint = Output(UInt(64.W))
  val online      = Output(Bool())

  val bind = Flipped(Valid(new Bundle {
    val task     = UInt(p.taskBits.W)
    val snapshot = new CommandSnapshot(tracking, pmps)
  }))

  val drained     = Output(Bool())
  val workDrained = Input(Bool())
  val halted      = Input(Bool())
}

/** The tile admits commands and joins their actual ROB, memory and fault lifetimes. */
class NpuEvents(b: GlobalConfig) extends Bundle {
  val command    = Flipped(Decoupled(new RoCCCommandBB(b.tile.xLen)))
  val response   = Decoupled(new RoCCResponseBB(b.tile.xLen))
  val allocation = Valid(new RobAllocation(b))
  val retired    = Output(UInt(b.frontend.rob_entries.W))
  val fault      = Valid(new RobFault(b.frontend.rob_entries))
  val busy       = Output(Bool())
  val interrupt  = Output(Bool())
  val footprints = Output(Vec(3, new Footprint(b)))
}

/** Real NPU/issuer boundary; translation, cache maintenance and task-group storage stay in Tile. */
class CLinkIO(
  b:                      GlobalConfig,
  ant:                    AntParams,
  tracking:               TrackingParams,
  bus:                    chi.Params
)(
  implicit val cpuParams: CpuParams)
    extends CLink {
  val ctrl      = new Control(ant, tracking, cpuParams.core.nPMPs)
  val local     = new LocalPort(ant)
  val admission = new AdmissionPorts(tracking, cpuParams.core.nPMPs, bus)
  val memory    = new AntMemoryPort(tracking, bus)
  val npu       = new NpuEvents(b)
  val shm       = new ShmPort(b)
  val mem       = new DmaPort(b.memDomain.dma_buswidth)

  val sharedHashes =
    if (b.sim.diffTest && b.memDomain.sharedEnable)
      Some(Input(Vec(SharedMemLayout.totalBank(b), new PhysicalBankHash(b))))
    else None

}
