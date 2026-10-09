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
import framework.system.core.rocket.{CpuParams, RoCCIO}
import framework.system.configloader.{AntTileCore, RocketTileCore, SharedStorageParams, TileCore, TileTopology}
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.core.{ControllerAdmission, RocketCLink}
import hier.core.rocket.Commands
import framework.system.core.accelerator.Admission
import framework.memdomain.frontend.mem.dma.DmaStatus
import framework.memdomain.backend.shared.{SharedMemLayout, SharedPhysicalPort, SharedPhysicalRequest}
import framework.memdomain.backend.MTraceDPI
import memcore.memory.mesh_shm.{MeshCoreAttachment, MeshSharedMem, MeshSharedMemParams}
import hier.tile.memory.{Composition, CoreInterrupts, CorePlacement, CoreRole}
import memcore.bus.chi.RequesterPort
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.preflight.{Params => PreparationParams}

/** A PB-defined Tile assembled explicitly from its CPU, cache, accelerator and bank IPs. */
@instantiable
class MainTilePlatform(p: TileParams) extends TileEndpoint(p.linkParams, true) {
  val coreTypes     = p.cores
  val cpuParameters = p.cpus
  val memory        = p.memory
  val l1            = p.l1
  val regions       = p.regions
  val tracking      = p.tracking
  val controlCores  = p.controls
  val axiParams     = p.axi
  val sharedStorage = p.sharedStorage
  val tiles         = p.tiles
  require(p.controller.isEmpty, "Main tile has no hidden task controller")
  require(coreTypes.size == cpuParameters.size)
  require(memory.agents == 2 * coreTypes.size)

  val cores = coreTypes.map {
    case core: RocketTileCore => core
    case _ => throw new IllegalArgumentException("This Tile requires Rocket CPUs")
  }

  // A CPU-only Tile (for example a main tile without Buckyball cores) has no banks or NPU DMA.
  val enabled = cores.indices.filter(i => cores(i).buckyball.isDefined)
  val base    = enabled.headOption.map(i => cores(i).buckyball.get)
  // Shared banks form the Tile bank mesh; otherwise every accelerator keeps its private banks.
  val shared  = base.exists(_.memDomain.sharedEnable)
  for (b <- base) {
    require(b.memDomain.nCores == cores.size)
    require(b.memDomain.computeCoreIds == enabled)
  }
  for {
    i    <- enabled
    base <- base
  } {
    val b = cores(i).buckyball.get
    require(b.tile.xLen == 64 && b.memDomain.nCores == cores.size && b.memDomain.sharedEnable == shared)
    require(b.memDomain.bankWidth == base.memDomain.bankWidth &&
      b.memDomain.dma_buswidth == base.memDomain.dma_buswidth &&
      b.memDomain.bankChannel == base.memDomain.bankChannel &&
      b.frontend.rob_entries == base.frontend.rob_entries)
  }
  val c = memory.chi
  val cp       = CpuMemParams(c, tagBits = 6)
  val prepared =
    PreparationParams(c, tracking.entries, tracking.maxRanges, tracking.idBits, beatBytes = axiParams.bytes)

  val placements = cores.indices.map { i =>
    CorePlacement(
      l1.copy(nodeId = i + 1),
      cpuParameters(i),
      CoreRole(compute = cores(i).buckyball.isDefined, scheduler = true)
    )
  }

  val controlCoreIds = cores.indices

  // TLink and NPU requests share the same bank network when main has an accelerator.
  val controllerMove = shared && cores.head.buckyball.isEmpty

  val banks: Option[Instance[BankNetwork]] = Option.when(shared)(
    Instantiate(new BankNetwork(
      base.get,
      enabled,
      useMesh = true,
      controllerMove = controllerMove,
      physicalPorts = if (sharedStorage.isDefined) 4 else 0
    ))
  )

  val mainTransfer = sharedStorage.map { geometry =>
    require(enabled.isEmpty || shared, "Main NPU and TLink require one shared bank network")
    val p        = T2TParams(geometry.bytes, geometry.bankBytes, tiles, axiParams)
    val control  = Instantiate(new TransferControl(p, localAccess = true, leaseAccess = banks.isDefined))
    val transfer = Instantiate(new T2TTransfer(p))
    val endpoint = Instantiate(new SharedAxiEndpoint(p))
    val physical = Wire(Vec(4, Flipped(new SharedPhysicalPort(128))))
    val pending  = RegInit(VecInit(Seq.fill(4)(false.B)))
    banks match {
      case Some(network) =>
        val b = base.get
        require(
          b.memDomain.bankWidth == geometry.bankBits &&
            b.memDomain.sharedBankEntries == geometry.bankEntries &&
            SharedMemLayout.totalBank(b) == geometry.banks,
          "Main NPU and TLink storage geometry must agree"
        )
        network.io.physical <> physical
        network.io.lease.get <> control.io.lease.get
        for ((port, index) <- physical.zipWithIndex) {
          when(port.request.fire) {
            assert(!pending(index), "Main SHM port accepts one outstanding request")
            pending(index) := true.B
          }
          when(port.response.fire) {
            assert(pending(index) && !port.response.bits.error, "Main SHM response has no owner or failed")
            pending(index) := false.B
          }
        }
      case None          =>
        val rows = math.ceil(math.sqrt(geometry.banks.toDouble)).toInt
        val shm  = Instantiate(new MeshSharedMem(MeshSharedMemParams(
          rows = rows,
          cols = (geometry.banks + rows - 1) / rows,
          entriesPerBank = geometry.bankEntries,
          dataBits = geometry.bankBits,
          tagBits = 8,
          cores = Seq(MeshCoreAttachment(Seq(0))),
          externalChannels = 4,
          visibleBanks = geometry.banks
        )))
        // Main has no private NPU endpoint. Its CPU management and T2T use physical SHM ports.
        for (channel <- shm.io.channels) {
          channel.request.valid  := false.B
          channel.request.bits   := 0.U.asTypeOf(channel.request.bits)
          channel.response.ready := true.B
        }
        shm.io.transferCommand.valid := false.B
        shm.io.transferCommand.bits     := 0.U.asTypeOf(shm.io.transferCommand.bits)
        shm.io.transferCompletion.ready := true.B
        for (local <- shm.io.localBanks) {
          local.request.ready  := false.B
          local.response.valid := false.B
          local.response.bits  := 0.U.asTypeOf(local.response.bits)
        }
        shm.io.bankWrites.foreach(_.ready := true.B)
        val bankBits = log2Ceil(geometry.bankBytes)
        for ((port, index) <- physical.zipWithIndex) {
          val channel = shm.io.external(index)
          val held    = Reg(new SharedPhysicalRequest(128))
          when(port.request.valid && !reset.asBool) {
            assert(
              port.request.bits.address(3, 0) === 0.U && port.request.bits.address < geometry.bytes.U,
              "Main SHM physical access is unaligned or exceeds capacity"
            )
          }
          channel.request.valid      := port.request.valid
          channel.request.bits       := 0.U.asTypeOf(channel.request.bits)
          channel.request.bits.bank  := port.request.bits.address >> bankBits
          channel.request.bits.tuser := Cat(port.request.bits.address(bankBits - 1, 4).pad(16), port.request.bits.write)
          channel.request.bits.data  := port.request.bits.data
          channel.request.bits.mask  := port.request.bits.mask
          channel.request.bits.tag   := 0.U
          channel.request.bits.tlast := true.B
          port.request.ready         := channel.request.ready
          port.response.valid        := channel.response.valid
          port.response.bits.data    := channel.response.bits.data
          port.response.bits.error   := channel.response.bits.error
          channel.response.ready     := port.response.ready
          when(port.request.fire) {
            assert(!pending(index), "Main SHM port accepts one outstanding request")
            pending(index) := true.B
            held           := port.request.bits
          }
          when(port.response.fire) {
            assert(pending(index) && !port.response.bits.error, "Main SHM response has no owner or failed")
            pending(index) := false.B
          }
          val trace = Module(new MTraceDPI)
          val value = Mux(held.write, held.data, port.response.bits.data)
          trace.io.clock      := clock
          trace.io.reset      := reset.asBool
          trace.io.hart_id    := tlink.executionIds(0)
          trace.io.channel    := (shm.io.channels.size + index).U
          trace.io.is_shared  := 1.U
          trace.io.is_write   := held.write.asUInt
          trace.io.rob_id     := 0.U
          trace.io.inst_id    := 0.U
          trace.io.vbank_id   := 0.U
          trace.io.group_id   := 0.U
          trace.io.pbank_id   := held.address >> bankBits
          trace.io.addr       := held.address(bankBits - 1, 4)
          trace.io.write_mask := Mux(held.write, held.mask, 0.U)
          trace.io.data_lo    := value(63, 0)
          trace.io.data_hi    := value(127, 64)
          trace.io.enable     := port.response.fire
        }
    }
    control.io.tileId := tlink.tileId
    transfer.io.command <> control.io.command
    control.io.completion <> transfer.io.completion
    physical(0) <> transfer.io.source
    physical(1) <> endpoint.io.read
    physical(2) <> endpoint.io.write
    physical(3) <> control.io.local.get
    tlink.t2t.get.tx <> transfer.io.mem
    endpoint.io.mem <> tlink.t2t.get.rx
    (control, endpoint, pending.asUInt.orR)
  }

  val tileMemory = Option.when(enabled.nonEmpty)(
    Instantiate(new memcore.bus.axi4.Interconnect(axiParams, enabled.size))
  )

  tileMemory match {
    case Some(fabric) => tlink.mem <> fabric.io.out
    case None         =>
      tlink.mem.aw.valid := false.B
      tlink.mem.aw.bits  := 0.U.asTypeOf(tlink.mem.aw.bits)
      tlink.mem.w.valid  := false.B
      tlink.mem.w.bits   := 0.U.asTypeOf(tlink.mem.w.bits)
      tlink.mem.ar.valid := false.B
      tlink.mem.ar.bits  := 0.U.asTypeOf(tlink.mem.ar.bits)
      tlink.mem.b.ready  := true.B
      tlink.mem.r.ready  := true.B
  }

  val composition: Instance[Composition] = Instantiate(new Composition(
    memory,
    placements,
    regions,
    tracking,
    Nil,
    Nil,
    controlCoreIds
  ))

  composition.io.hartIds     := tlink.hartIds
  composition.io.resetVector := tlink.resetVector
  composition.io.time        := tlink.time
  composition.io.interrupts  := tlink.interrupts
  tlink.uncachedRequest <> composition.io.uncachedRequest
  composition.io.uncachedResponse <> tlink.uncachedResponse
  locally {
    val memorySystem = Instantiate(new hier.tile.memory.Memory(memory.copy(agents = 2 * controlCores)))
    for (lane <- 0 until 2 * cores.size) {
      val localNode   = lane + 1
      val globalIndex = lane % cores.size + (if (lane >= cores.size) controlCores else 0)
      val source      = composition.io.control(lane)
      val target      = memorySystem.io.coherent(globalIndex)
      target <> source
      target.req.bits.srcId   := (globalIndex + 1).U
      target.txRsp.bits.srcId := (globalIndex + 1).U
      target.txDat.bits.srcId := (globalIndex + 1).U
      source.rxRsp.bits.tgtId := localNode.U
      source.rxDat.bits.tgtId := localNode.U
    }
    val remoteCount = controlCores - cores.size
    for (lane <- 0 until 2 * remoteCount) {
      val globalIndex = cores.size + lane % remoteCount + (if (lane >= remoteCount) controlCores else 0)
      memorySystem.io.coherent(globalIndex) <> tlink.remoteControl(lane)
    }
    tlink.backingRequest <> memorySystem.io.backingReq
    memorySystem.io.backingResp <> tlink.backingResponse
  }
  tlink.retired              := composition.io.retired
  tlink.retiredPc            := composition.io.retiredPc
  tlink.trapped              := composition.io.trapped
  tlink.trapCause            := composition.io.trapCause
  tlink.trapValue            := composition.io.trapValue
  tlink.trapPc               := composition.io.trapPc
  banks.foreach(_.io.hartIds := tlink.executionIds)
  tlink.failure              := 0.U.asTypeOf(tlink.failure)

  def connectTask(i: Int, task: RoCCIO, npuIdle: Bool = true.B): Unit = {
    val target = composition.io.taskControl(i)
    if (i == 0 && mainTransfer.isDefined) {
      val (control, _, _) = mainTransfer.get
      val transfer        = task.cmd.bits.funct >= 13.U
      val accessesStorage = task.cmd.bits.funct === 14.U ||
        (task.cmd.bits.funct >= 16.U && task.cmd.bits.funct <= 19.U)
      val transferReady   = !accessesStorage || npuIdle
      val rd              = Reg(UInt(5.W))
      control.io.request.valid          := task.cmd.valid && transfer && transferReady
      control.io.request.bits.operation := task.cmd.bits.funct
      control.io.request.bits.context   := task.cmd.bits.rs1Data(63, 32)
      control.io.request.bits.field     := task.cmd.bits.rs1Data(31, 0)
      control.io.request.bits.data      := task.cmd.bits.rs2Data
      target.cmd.valid                  := task.cmd.valid && !transfer
      target.cmd.bits                   := task.cmd.bits
      target.exception                  := task.exception
      task.cmd.ready                    := Mux(transfer, control.io.request.ready && transferReady, target.cmd.ready)
      when(task.cmd.fire) {
        assert(task.cmd.bits.opcode === "h2b".U && task.cmd.bits.funct3 === 7.U, "Invalid main management instruction")
        rd := task.cmd.bits.rd
      }
      assert(!(control.io.reply.valid && target.resp.valid), "Concurrent main management responses")
      task.resp.valid                   := control.io.reply.valid || target.resp.valid
      task.resp.bits.rd                 := Mux(control.io.reply.valid, rd, target.resp.bits.rd)
      task.resp.bits.data               := Mux(control.io.reply.valid, control.io.reply.bits, target.resp.bits.data)
      control.io.reply.ready            := task.resp.ready
      target.resp.ready                 := task.resp.ready
      task.busy                         := !control.io.idle || target.busy
      task.interrupt                    := target.interrupt
    } else task <> target
  }

  @public val coreLinks = IO(MixedVec(cores.indices.map(i => Flipped(new RocketCLink(p.rocket(i))))))
  for (i <- cores.indices) {
    composition.io.cores(i) <> coreLinks(i).cpu
    cores(i).buckyball match {
      case Some(b) =>
        val accelerator = coreLinks(i).accelerator.get
        val admission: Instance[Admission] =
          Instantiate(new Admission(b, tracking, prepared, regions, axiParams)(cpuParameters(i)))
        admission.io.core <> composition.io.admission(i)
        connectTask(i, admission.io.task, !accelerator.npu.busy)
        if (i == 0) composition.io.controllerSatp := admission.io.taskSatp
        if (i == 0 && mainTransfer.isDefined) {
          // A younger NPU command cannot pass an active local storage-management command.
          val storageIdle = mainTransfer.get._1.io.idle
          accelerator.npu.command.valid := admission.io.npuCommand.valid && storageIdle
          accelerator.npu.command.bits  := admission.io.npuCommand.bits
          admission.io.npuCommand.ready := accelerator.npu.command.ready && storageIdle
        } else accelerator.npu.command <> admission.io.npuCommand
        admission.io.npuResponse <> accelerator.npu.response
        admission.io.allocation.valid             := accelerator.npu.allocation.valid
        admission.io.allocation.bits              := accelerator.npu.allocation.bits.rob_id
        admission.io.retired                      := accelerator.npu.retired
        admission.io.npuFault                     := accelerator.npu.fault
        admission.io.npuBusy                      := accelerator.npu.busy
        admission.io.npuInterrupt                 := accelerator.npu.interrupt
        admission.io.footprints                   := accelerator.npu.footprints
        admission.io.dma <> accelerator.mem
        val bankEndpoint = enabled.indexOf(i)
        tileMemory.get.io.in(bankEndpoint) <> admission.io.axi
        accelerator.hartId     := tlink.executionIds(i)
        accelerator.shmOwner   := tlink.executionIds(0)
        banks match {
          case Some(network) =>
            val port = network.io.compute(bankEndpoint)
            port.requests <> accelerator.shm.requests
            port.move <> accelerator.shm.move
            accelerator.shm.local <> port.local
            port.config <> accelerator.shm.config
            port.queryValid                    := accelerator.shm.queryValid
            port.queryVbank                    := accelerator.shm.queryVbank
            accelerator.shm.queryGroups        := port.queryGroups
            port.barrierArrive                 := accelerator.shm.barrierArrive
            accelerator.shm.barrierRelease     := port.barrierRelease
            accelerator.sharedHashes.foreach(_ := network.io.bankHashes.get)
          case None          =>
            // Private banks: no shared channels exist, and mesh moves and barriers stay idle.
            accelerator.shm.move.command.ready    := false.B
            accelerator.shm.move.completion.valid := false.B
            accelerator.shm.move.completion.bits  := false.B
            accelerator.shm.local.request.valid   := false.B
            accelerator.shm.local.request.bits    := 0.U.asTypeOf(accelerator.shm.local.request.bits)
            accelerator.shm.local.response.ready  := true.B
            accelerator.shm.config.ready          := false.B
            accelerator.shm.queryGroups           := 0.U
            accelerator.shm.barrierRelease        := false.B
        }
        tlink.failure(i).valid := admission.io.halted
        tlink.failure(i).bits  := admission.io.fault
        tlink.workDrained(i)   := admission.io.workDrained &&
          (if (i == 0) mainTransfer.map { case (control, endpoint, storageBusy) =>
             control.io.idle && !endpoint.io.busy && !storageBusy
           }.getOrElse(true.B)
           else true.B)
      case None    =>
        val admission: Instance[ControllerAdmission] =
          Instantiate(new ControllerAdmission(tracking, c, moves = i == 0 && controllerMove)(cpuParameters(i)))
        admission.io.core <> composition.io.admission(i)
        connectTask(i, admission.io.task)
        if (i == 0) composition.io.controllerSatp := admission.io.taskSatp
        if (i == 0 && controllerMove) {
          banks.get.io.controllerMvover.get <> admission.io.move
        } else {
          admission.io.move.command.ready    := false.B
          admission.io.move.completion.valid := false.B
          admission.io.move.completion.bits  := false.B
        }
        tlink.failure(i)                          := admission.io.moveFault
        tlink.workDrained(i)                      := composition.io.admission(i).outstanding === 0.U &&
          (if (i == 0) mainTransfer.map { case (control, endpoint, storageBusy) =>
             control.io.idle && !endpoint.io.busy && !storageBusy
           }.getOrElse(true.B)
           else true.B)
    }
  }
}
