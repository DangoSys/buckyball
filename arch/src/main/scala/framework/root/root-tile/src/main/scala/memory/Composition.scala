package hier.tile.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.CpuParams
import hier.core.rocket.{AdmissionPorts, Commands, CpuCLink}
import hier.tile.TaskController
import framework.system.core.rocket.RoCCIO
import memcore.bus.chi.RequesterPort
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
  l1:   RnfParams,
  cpu:  CpuParams,
  role: CoreRole)

class CoreInterrupts extends Bundle {
  val timer              = Bool()
  val software           = Bool()
  val external           = Bool()
  val supervisorExternal = Bool()
}

/** Explicit CPU and task assembly; the Chip owner supplies each Core admission. */
@instantiable
class Composition(
  memory:         CoherenceParams,
  placements:     Seq[CorePlacement],
  regions:        Seq[PhysicalRegion],
  tracking:       TrackingParams,
  workerCoreIds:  Seq[Int],
  signatures:     Seq[BigInt],
  controlCoreIds: Seq[Int] = Nil)
    extends Module {
  val localCoreIds = placements.indices.filterNot(controlCoreIds.contains)
  require(controlCoreIds.distinct.size == controlCoreIds.size)
  require(controlCoreIds.forall(placements.indices.contains))
  // Control requesters leave the tile; the remaining cores use the local Home.
  require(placements.nonEmpty && 2 * placements.size == memory.agents)
  require(workerCoreIds.size == signatures.size && signatures.forall(s => s >= 0 && s < (BigInt(1) << 64)))
  require(
    workerCoreIds.forall(i => i > 0 && i < placements.size),
    "Task worker IDs must explicitly name non-controller placements"
  )
  require(placements.head.role.scheduler, "TaskController port zero must be a scheduler-capable placement")
  val c            = memory.chi
  val cp           = CpuMemParams(c, tagBits = 6)
  require(tracking.addressBits == c.addressBits)
  placements.zipWithIndex.foreach { case (entry, index) =>
    require(entry.l1.chi == c && entry.l1.homeId == memory.homeId && entry.l1.homeCount == 1)
    // Memory/Fabric routes node i+1. This is a CHI node contract, never a hart-ID derivation.
    require(entry.l1.nodeId == index + 1, "Core CHI node must match its Memory requester port")
  }

  @public
  val io = IO(new Bundle {

    // Caller routes opcode 0x2b commands and supplies their captured controller SATP.
    val cores = MixedVec(placements.zipWithIndex.map { case (entry, index) =>
      val controlIndex = controlCoreIds.indexOf(index)
      val localIndex   = localCoreIds.indexOf(index)
      val node         = if (controlIndex >= 0) controlIndex + 1 else localIndex + 1
      new CpuCLink(entry.l1.copy(nodeId = node), Some(Commands(entry.role.compute, entry.role.scheduler, tracking)))(
        entry.cpu
      )
    }.map(Flipped(_)))

    val taskControl      = Vec(placements.size, new RoCCIO(64))
    val controllerSatp   = Input(UInt(64.W))
    val hartIds          = Input(Vec(placements.size, UInt(64.W)))
    val resetVector      = Input(Vec(placements.size, UInt(64.W)))
    val time             = Input(UInt(64.W))
    val interrupts       = Input(Vec(placements.size, new CoreInterrupts))
    val admission        =
      MixedVec(placements.map(entry => new AdmissionPorts(tracking, entry.cpu.core.nPMPs, c)(entry.cpu)))
    val uncachedRequest  = Vec(placements.size, Decoupled(new UncachedRequest(cp)))
    val uncachedResponse = Vec(placements.size, Flipped(Decoupled(new UncachedResponse(cp))))
    val control          = Vec(2 * controlCoreIds.size, new RequesterPort(c))
    val backingReq       = Vec(if (localCoreIds.nonEmpty) 1 else 0, Decoupled(new LineRequest(c)))
    val backingResp      = Vec(if (localCoreIds.nonEmpty) 1 else 0, Flipped(Decoupled(new LineResponse(c))))
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

  val memorySystem = Option.when(localCoreIds.nonEmpty)(
    Instantiate(new Memory(memory.copy(agents = 2 * localCoreIds.size)))
  )

  io.homeOutstanding := 0.U
  memorySystem.foreach { m =>
    io.backingReq <> m.io.backingReq
    m.io.backingResp <> io.backingResp
    io.homeOutstanding := m.io.homeOutstanding
  }

  for (index <- placements.indices) {
    val controlIndex = controlCoreIds.indexOf(index)
    val localIndex   = localCoreIds.indexOf(index)
    val core         = io.cores(index)
    core.hartId                      := io.hartIds(index)
    core.resetVector                 := io.resetVector(index)
    core.time                        := io.time
    core.timerInterrupt              := io.interrupts(index).timer
    core.softwareInterrupt           := io.interrupts(index).software
    core.externalInterrupt           := io.interrupts(index).external
    core.supervisorExternalInterrupt := io.interrupts(index).supervisorExternal
    io.admission(index) <> core.admission.get
    io.uncachedRequest(index) <> core.uncachedRequest
    core.uncachedResponse <> io.uncachedResponse(index)
    if (controlIndex >= 0) {
      io.control(controlIndex) <> core.chi
      io.control(controlCoreIds.size + controlIndex) <> core.instructionChi
    } else {
      memorySystem.get.io.coherent(localIndex) <> core.chi
      memorySystem.get.io.coherent(localCoreIds.size + localIndex) <> core.instructionChi
    }
    io.retired(index)                := core.retired
    io.retiredPc(index)              := core.retiredPc
    io.trapped(index)                := core.trapped
    io.trapCause(index)              := core.trapCause
    io.trapValue(index)              := core.trapValue
    io.trapPc(index)                 := core.trapPc
  }
}
