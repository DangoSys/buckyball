package framework.system.tile

import chisel3._
import framework.system.tile.{BankNetwork, TileEndpoint, TileParams}
import framework.system.tile.tlink.{
  HasTLink,
  TLinkIO,
  T2TParams,
  T2TTransfer,
  SharedAxiEndpoint,
  Control => TransferControl
}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ant.Service
import framework.system.configloader.{AntTileCore, RocketTileCore}
import framework.system.core.{ControllerAdmission, RocketCLink}
import hier.core.rocket.Commands
import framework.system.core.rocket.CpuParams
import framework.system.core.accelerator.{Admission, AntMemory}
import framework.memdomain.frontend.mem.dma.DmaStatus
import framework.memdomain.backend.shared.SharedMemLayout
import hier.tile.memory.{Composition, CorePlacement, CoreRole}
import memcore.bus.axi4
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.preflight.{Params => PreparationParams}

/** One coherent Linux controller, local Ant execution contexts, and their NPU endpoints. */
@instantiable
class ComputeTilePlatform(p: TileParams) extends TileEndpoint(p.linkParams, false) {
  require(!p.main && p.controller.contains(0) && p.cpus.size == 1)

  val controller = p.cores.head match {
    case core: RocketTileCore => core
    case _ => throw new IllegalArgumentException("Compute controller must be Rocket")
  }

  val workers = p.cores.tail.map {
    case core: AntTileCore => core
    case _ => throw new IllegalArgumentException("Compute workers must be Ant")
  }

  val signatures = p.signatures.tail
  val cpu        = p.cpus.head
  val memory     = p.memory
  val l1         = p.l1
  val regions    = p.regions
  val tracking   = p.tracking
  val axiParams  = p.axi
  val tiles      = p.tiles
  require(controller.buckyball.isEmpty && workers.nonEmpty && workers.size == signatures.size)
  val local      = workers.head.local
  require(workers.forall(_.local == local), "Ant local storage geometry must be homogeneous within a tile")
  val base       = workers.head.buckyball
  val enabled    = 1 to workers.size
  require(workers.forall(w =>
    w.buckyball.memDomain.sharedEnable &&
      w.buckyball.memDomain.computeCoreIds == enabled && w.buckyball.memDomain.nCores == workers.size + 1
  ))
  val c          = memory.chi
  val prepared   =
    PreparationParams(c, tracking.entries, tracking.maxRanges, tracking.idBits, beatBytes = axiParams.bytes)

  val tileMemory = Instantiate(new axi4.Interconnect(axiParams, workers.size))
  tlink.mem <> tileMemory.io.out

  val composition = Instantiate(new Composition(
    memory.copy(agents = 2),
    Seq(CorePlacement(l1.copy(nodeId = 1), cpu, CoreRole.Controller)),
    regions,
    tracking,
    Nil,
    Nil,
    Seq(0)
  ))

  @public val controllerLink = IO(Flipped(new RocketCLink(p.rocket(0))))

  @public val workerLinks = IO(MixedVec((1 until p.cores.size).map { i =>
    val params = p.ant(i)
    Flipped(new framework.system.core.clink.CLinkIO(
      params.buckyball,
      params.local,
      params.tracking,
      params.bus
    )(params.cpu))
  }))

  composition.io.cores(0) <> controllerLink.cpu

  val controllerAdmission = Instantiate(new ControllerAdmission(tracking, c, moves = true)(cpu))
  val service             = Instantiate(new Service(local, workers.size))
  val broker              = Instantiate(new AntMemory(workers.size, tracking, c))
  val banks               = Instantiate(new BankNetwork(base, enabled, useMesh = true, controllerMove = true, physicalPorts = 3))

  val transferParams = T2TParams(
    sharedBytes = BigInt(SharedMemLayout.totalBank(base)) * base.memDomain.sharedBankEntries * axiParams.bytes,
    bankBytes = base.memDomain.sharedBankEntries * axiParams.bytes,
    tiles = tiles,
    axi = axiParams
  )

  val transferControl = Instantiate(new TransferControl(transferParams))
  val transfer        = Instantiate(new T2TTransfer(transferParams))
  val sharedEndpoint  = Instantiate(new SharedAxiEndpoint(transferParams))
  transferControl.io.tileId  := tlink.tileId
  transfer.io.command <> transferControl.io.command
  transferControl.io.completion <> transfer.io.completion
  banks.io.physical(0) <> transfer.io.source
  banks.io.physical(1) <> sharedEndpoint.io.read
  banks.io.physical(2) <> sharedEndpoint.io.write
  banks.io.lease.get <> transferControl.io.lease.get
  tlink.t2t.get.tx <> transfer.io.mem
  sharedEndpoint.io.mem <> tlink.t2t.get.rx
  composition.io.hartIds     := tlink.hartIds
  composition.io.resetVector := tlink.resetVector
  composition.io.time        := tlink.time
  composition.io.interrupts  := tlink.interrupts
  tlink.control <> composition.io.control
  tlink.uncachedRequest <> composition.io.uncachedRequest
  composition.io.uncachedResponse <> tlink.uncachedResponse
  tlink.retired              := composition.io.retired
  tlink.retiredPc            := composition.io.retiredPc
  tlink.trapped              := composition.io.trapped
  tlink.trapCause            := composition.io.trapCause
  tlink.trapValue            := composition.io.trapValue
  tlink.trapPc               := composition.io.trapPc

  // No hidden CPU polls the legacy task port. Only the controller drives this service.
  composition.io.controllerSatp := 0.U
  for (port <- composition.io.taskControl) {
    port.cmd.valid  := false.B
    port.cmd.bits   := 0.U.asTypeOf(port.cmd.bits)
    port.resp.ready := true.B
    port.exception  := false.B
  }
  val core = composition.io.admission(0)
  controllerAdmission.io.core <> core
  val task       = controllerAdmission.io.task
  val rd         = Reg(UInt(5.W))
  val isTransfer = task.cmd.bits.funct >= 13.U
  transferControl.io.request.bits.operation := task.cmd.bits.funct
  transferControl.io.request.bits.context   := task.cmd.bits.rs1Data(63, 32)
  transferControl.io.request.bits.field     := task.cmd.bits.rs1Data(31, 0)
  transferControl.io.request.bits.data      := task.cmd.bits.rs2Data
  transferControl.io.request.valid          := task.cmd.valid && isTransfer
  service.io.request.valid                  := task.cmd.valid && !isTransfer
  service.io.request.bits.operation         := task.cmd.bits.funct
  service.io.request.bits.context           := task.cmd.bits.rs1Data(63, 32)
  service.io.request.bits.field             := task.cmd.bits.rs1Data(31, 0)
  service.io.request.bits.data              := task.cmd.bits.rs2Data
  task.cmd.ready                            := Mux(isTransfer, transferControl.io.request.ready, service.io.request.ready)
  when(task.cmd.fire) {
    assert(task.cmd.bits.opcode === "h2b".U && task.cmd.bits.funct3 === 7.U, "Invalid Ant management instruction")
    rd := task.cmd.bits.rd
  }
  task.resp.valid                           := service.io.reply.valid || transferControl.io.reply.valid
  task.resp.bits.rd                         := rd
  task.resp.bits.data                       := Mux(transferControl.io.reply.valid, transferControl.io.reply.bits, service.io.reply.bits)
  transferControl.io.reply.ready            := task.resp.ready
  assert(!(service.io.reply.valid && transferControl.io.reply.valid), "Concurrent tile management responses")
  service.io.reply.ready                    := task.resp.ready
  task.busy                                 := !service.io.idle || !transferControl.io.idle
  task.interrupt                            := false.B
  banks.io.hartIds                          := tlink.executionIds
  banks.io.controllerMvover.get <> controllerAdmission.io.move

  broker.io.controller.cpuQuery := core.cpuQuery
  core.cpuAllow                 := controllerAdmission.io.core.cpuAllow && broker.io.controller.cpuAllow
  core.cpuProbeAllow            := controllerAdmission.io.core.cpuProbeAllow && broker.io.controller.cpuProbeAllow
  core.maintenance <> broker.io.controller.maintenance
  broker.io.controller.maintained <> core.maintained
  core.pteRequest <> broker.io.controller.pteRequest
  broker.io.controller.pteResponse <> core.pteResponse

  val failures = Wire(Vec(workers.size, Valid(new DmaStatus)))
  val drained  = Wire(Vec(workers.size, Bool()))
  for ((worker, i) <- workers.zipWithIndex) {
    val b         = worker.buckyball
    val link      = workerLinks(i)
    val admission = Instantiate(new Admission(b, tracking, prepared, regions, axiParams)(cpu))
    link.ctrl.bind.valid          := service.io.launched(i).valid
    link.ctrl.bind.bits.task      := service.io.launched(i).bits.task
    link.ctrl.bind.bits.snapshot  := controllerAdmission.io.taskContext
    service.io.locals(i) <> link.local
    link.ctrl.workDrained         := admission.io.workDrained
    link.ctrl.halted              := admission.io.halted
    drained(i)                    := link.ctrl.drained
    service.io.signatures(i)      := link.ctrl.fingerprint
    service.io.online(i)          := link.ctrl.online
    link.admission <> admission.io.core
    link.memory <> broker.io.clients(i)
    admission.io.task.cmd.ready   := false.B
    admission.io.task.resp.valid  := false.B
    admission.io.task.resp.bits   := 0.U.asTypeOf(admission.io.task.resp.bits)
    admission.io.task.busy        := false.B
    admission.io.task.interrupt   := false.B
    link.npu.command <> admission.io.npuCommand
    admission.io.npuResponse <> link.npu.response
    admission.io.allocation.valid := link.npu.allocation.valid
    admission.io.allocation.bits  := link.npu.allocation.bits.rob_id
    admission.io.retired          := link.npu.retired
    admission.io.npuFault         := link.npu.fault
    admission.io.npuBusy          := link.npu.busy
    admission.io.npuInterrupt     := link.npu.interrupt
    admission.io.footprints       := link.npu.footprints
    admission.io.dma <> link.mem
    tileMemory.io.in(i) <> admission.io.axi
    failures(i).valid             := admission.io.halted
    failures(i).bits              := admission.io.fault
    link.ctrl.hartId              := tlink.executionIds(i + 1)
    link.ctrl.shmOwner            := tlink.executionIds(0)
    banks.io.compute(i) <> link.shm
    link.sharedHashes.foreach(_   := banks.io.bankHashes.get)
  }
  tlink.failure(0).valid := failures.map(_.valid).reduce(_ || _) || controllerAdmission.io.moveFault.valid
  tlink.failure(0).bits := Mux(
    controllerAdmission.io.moveFault.valid,
    controllerAdmission.io.moveFault.bits,
    PriorityMux(failures.map(f => f.valid -> f.bits))
  )
  tlink.workDrained(
    0
  )                     := service.io.idle && transferControl.io.idle && !sharedEndpoint.io.busy && drained.asUInt.andR && core.outstanding === 0.U
}
