package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.CpuParams
import hier.core.rocket.{AdmissionPorts, Commands, Core}
import hier.tile.TaskController
import framework.system.core.rocket.RoCCIO
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{Params => TrackingParams}

/** Roles describe command capabilities; a scheduler may also have compute capability. */
case class CoreRole(compute: Boolean, scheduler: Boolean) {
  require(compute || scheduler)
}

object CoreRole {
  val Controller = CoreRole(compute = false, scheduler = true)
  val Compute    = CoreRole(compute = true, scheduler = false)
}

case class CorePlacement(
  hartId: BigInt,
  l1:     RnfParams,
  cpu:    CpuParams,
  role:   CoreRole)

class CoreInterrupts extends Bundle {
  val timer              = Bool()
  val software           = Bool()
  val external           = Bool()
  val supervisorExternal = Bool()
}

/** Explicit CPU and task assembly; the Chip owner supplies each Core admission. */
@instantiable
class Composition(
  memory:        CoherenceParams,
  placements:    Seq[CorePlacement],
  regions:       Seq[PhysicalRegion],
  tracking:      TrackingParams,
  workerCoreIds: Seq[Int],
  signatures:    Seq[BigInt])
    extends Module {
  // Home requester i is placement i's data L1; requester n + i is its instruction L1.
  require(placements.nonEmpty && 2 * placements.size == memory.agents)
  require(placements.map(_.hartId).distinct.size == placements.size, "Tile hart IDs must be explicit and unique")
  require(workerCoreIds.size == signatures.size && signatures.forall(s => s >= 0 && s < (BigInt(1) << 64)))
  require(
    workerCoreIds.forall(i => i > 0 && i < placements.size),
    "Task worker IDs must explicitly name non-controller placements"
  )
  require(placements.head.role.scheduler, "TaskController port zero must be a scheduler-capable placement")
  private val c  = memory.chi
  private val cp = CpuMemParams(c, tagBits = 6)
  require(tracking.addressBits == c.addressBits)
  placements.zipWithIndex.foreach { case (entry, index) =>
    require(
      entry.hartId >= 0 && entry.hartId < (BigInt(1) << entry.cpu.hartIdBits),
      "Hart ID exceeds the selected Core's width"
    )
    require(entry.l1.chi == c && entry.l1.homeId == memory.homeId && entry.l1.homeCount == 1)
    // Memory/Fabric routes node i+1. This is a CHI node contract, never a hart-ID derivation.
    require(entry.l1.nodeId == index + 1, "Core CHI node must match its Memory requester port")
  }

  @public
  val io = IO(new Bundle {
    // Caller routes opcode 0x2b commands and supplies their captured controller SATP.
    val taskControl      = Vec(placements.size, new RoCCIO(64))
    val controllerSatp   = Input(UInt(64.W))
    val resetVector      = Input(Vec(placements.size, UInt(64.W)))
    val time             = Input(UInt(64.W))
    val interrupts       = Input(Vec(placements.size, new CoreInterrupts))
    val admission        =
      MixedVec(placements.map(entry => new AdmissionPorts(tracking, entry.cpu.core.nPMPs, c)(entry.cpu)))
    val uncachedRequest  = Vec(placements.size, Decoupled(new UncachedRequest(cp)))
    val uncachedResponse = Vec(placements.size, Flipped(Decoupled(new UncachedResponse(cp))))
    val backingReq       = Vec(1, Decoupled(new LineRequest(c)))
    val backingResp      = Vec(1, Flipped(Decoupled(new LineResponse(c))))
    val homeOutstanding  = Output(UInt(log2Ceil(memory.mshrEntries + 1).W))
    val retired          = Output(Vec(placements.size, Bool()))
    val retiredPc        = Output(Vec(placements.size, UInt(64.W)))
    val trapped          = Output(Vec(placements.size, Bool()))
    val trapCause        = Output(Vec(placements.size, UInt(64.W)))
    val trapValue        = Output(Vec(placements.size, UInt(64.W)))
    val trapPc           = Output(Vec(placements.size, UInt(64.W)))
  })

  if (workerCoreIds.nonEmpty) {
    val taskController: Instance[TaskController] =
      Instantiate(new TaskController(workerCoreIds, signatures, placements.size))
    taskController.io.ports <> io.taskControl
    taskController.io.satp := io.controllerSatp
  } else {
    // A Tile without task workers reports zero workers (op 3) and an invalid index (2) otherwise.
    for (port <- io.taskControl) {
      val pending  = RegInit(false.B)
      val rd       = Reg(UInt(5.W))
      val response = Reg(UInt(64.W))
      port.cmd.ready               := !pending
      port.resp.valid              := pending
      port.resp.bits.rd            := rd
      port.resp.bits.data          := response
      port.busy                    := pending
      port.interrupt               := false.B
      when(port.cmd.fire) {
        pending  := true.B
        rd       := port.cmd.bits.rd
        response := Mux(port.cmd.bits.funct === 3.U, 0.U, 2.U)
      }
      when(port.resp.fire)(pending := false.B)
    }
  }
  for (port <- io.taskControl) {
    when(port.cmd.valid) {
      assert(port.cmd.bits.opcode === "h2b".U, "Composition task port requires an opcode 0x2b command")
    }
  }

  val memorySystem: Instance[Memory] = Instantiate(new Memory(memory))
  io.backingReq <> memorySystem.io.backingReq
  memorySystem.io.backingResp <> io.backingResp
  io.homeOutstanding := memorySystem.io.homeOutstanding

  for ((entry, index) <- placements.zipWithIndex) {
    val commands = Commands(entry.role.compute, entry.role.scheduler, tracking)
    val core: Instance[Core] = Instantiate(
      new Core(entry.l1, entry.l1.copy(nodeId = placements.size + index + 1), regions, Some(commands))(entry.cpu)
    )
    core.io.hartId                      := entry.hartId.U
    core.io.resetVector                 := io.resetVector(index)
    core.io.time                        := io.time
    core.io.timerInterrupt              := io.interrupts(index).timer
    core.io.softwareInterrupt           := io.interrupts(index).software
    core.io.externalInterrupt           := io.interrupts(index).external
    core.io.supervisorExternalInterrupt := io.interrupts(index).supervisorExternal
    io.admission(index) <> core.io.admission.get
    io.uncachedRequest(index) <> core.io.uncachedRequest
    core.io.uncachedResponse <> io.uncachedResponse(index)
    memorySystem.io.coherent(index) <> core.io.chi
    memorySystem.io.coherent(placements.size + index) <> core.io.instructionChi
    io.retired(index)                   := core.io.retired
    io.retiredPc(index)                 := core.io.retiredPc
    io.trapped(index)                   := core.io.trapped
    io.trapCause(index)                 := core.io.trapCause
    io.trapValue(index)                 := core.io.trapValue
    io.trapPc(index)                    := core.io.trapPc
  }
}
