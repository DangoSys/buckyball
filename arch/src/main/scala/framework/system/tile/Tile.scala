package framework.system.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.CpuParams
import framework.system.configloader.{RocketTileCore, TileTopology}
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.core.ControllerAdmission
import framework.system.core.accelerator.{Admission, BuckyballAccelerator}
import framework.memdomain.frontend.mem.dma.DmaStatus
import hier.tile.memory.{Composition, CoreInterrupts, CorePlacement, CoreRole}
import memcore.bus.chi.rnf.RnfParams
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.preflight.{Params => PreparationParams}

object Tile {

  /** CPU metadata comes from the same PB as the accelerator and physical hart placement. */
  def cpuParameters(topology: TileTopology, maxHartId: Int, physicalAddressBits: Int): Seq[CpuParams] = {
    require(maxHartId >= topology.hartIds.max)
    val t = topology.param
    require(
      physicalAddressBits > t.pgIdxBits && physicalAddressBits <= t.paddrBits,
      "Platform physical address width must fit the PB address capability"
    )
    topology.cores.map {
      case RocketTileCore(cpu, _) =>
        require(
          !cpu.useVM || physicalAddressBits < t.vaddrBits,
          "Rocket requires platform physical addresses narrower than its virtual address format"
        )
        val core = RocketCpuParam.toRocketCoreParams(cpu, t.xLen, t.pgLevels).copy(
          useUser = cpu.useVM,
          useSupervisor = cpu.useVM,
          useHypervisor = false,
          useDebug = false,
          nPMPs = t.nPMPs,
          haveCease = false,
          haveSimTimeout = false
        )
        CpuParams(
          core = core,
          dcache = Some(RocketCpuParam.toDCacheParams(cpu, t.coreDataBytes * 8, 64)),
          icache = Some(RocketCpuParam.toICacheParams(cpu, t.coreDataBytes * 8, 64)),
          btb = RocketCpuParam.toBTBParams(cpu),
          physicalAddressBits = physicalAddressBits,
          hartIdBits = math.max(1, log2Ceil(maxHartId + 1)),
          beatBytes = t.coreDataBytes
        )
      case _                      => throw new IllegalArgumentException("This Tile requires Rocket CPUs")
    }
  }

}

/** A PB-defined Tile assembled explicitly from its CPU, cache, accelerator and bank IPs. */
@instantiable
class Tile(
  topology:      TileTopology,
  cpuParameters: Seq[CpuParams],
  memory:        CoherenceParams,
  l1:            RnfParams,
  regions:       Seq[PhysicalRegion],
  tracking:      TrackingParams,
  axiParams:     memcore.bus.axi4.Params)
    extends Module {
  require(topology.controller.forall(_ == 0), "A Tile task controller must be its slot zero")
  require(topology.cores.size == cpuParameters.size && topology.cores.size == topology.hartIds.size)
  require(memory.agents == 2 * topology.cores.size)
  require(topology.controller.isEmpty || topology.cores.size == topology.signatures.size)

  val cores = topology.cores.map {
    case core: RocketTileCore => core
    case _ => throw new IllegalArgumentException("This Tile requires Rocket CPUs")
  }

  // A CPU-only Tile (for example a main tile without Buckyball cores) has no banks or NPU DMA.
  val enabled        = cores.indices.filter(i => cores(i).buckyball.isDefined)
  val base           = enabled.headOption.map(i => cores(i).buckyball.get)
  // Shared banks form the Tile bank mesh; otherwise every accelerator keeps its private banks.
  private val shared = base.exists(_.memDomain.sharedEnable)
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
  private val c = memory.chi
  private val cp       = CpuMemParams(c, tagBits = 6)
  private val prepared =
    PreparationParams(c, tracking.entries, tracking.maxRanges, tracking.idBits, beatBytes = axiParams.bytes)

  private val placements = cores.indices.map { i =>
    CorePlacement(
      topology.hartIds(i),
      l1.copy(nodeId = i + 1),
      cpuParameters(i),
      CoreRole(compute = cores(i).buckyball.isDefined, scheduler = true)
    )
  }

  private val workers = topology.controller.map(c => cores.indices.filter(_ != c)).getOrElse(Nil)

  @public val io = IO(new Bundle {
    val resetVector      = Input(Vec(cores.size, UInt(64.W)))
    val interrupts       = Input(Vec(cores.size, new CoreInterrupts))
    val uncachedRequest  = Vec(cores.size, Decoupled(new UncachedRequest(cp)))
    val uncachedResponse = Vec(cores.size, Flipped(Decoupled(new UncachedResponse(cp))))
    // CPU L2 backing and NPU AXI data are independent paths.
    val backingRequest   = Vec(1, Decoupled(new LineRequest(c)))
    val backingResponse  = Vec(1, Flipped(Decoupled(new LineResponse(c))))
    val dma              = Vec(enabled.size, new memcore.bus.axi4.Port(axiParams))
    val failure          = Output(Vec(cores.size, Valid(new DmaStatus)))
    val workDrained      = Output(Vec(cores.size, Bool()))
    val retired          = Output(Vec(cores.size, Bool()))
    val retiredPc        = Output(Vec(cores.size, UInt(64.W)))
    val trapped          = Output(Vec(cores.size, Bool()))
    val trapCause        = Output(Vec(cores.size, UInt(64.W)))
    val trapValue        = Output(Vec(cores.size, UInt(64.W)))
    val trapPc           = Output(Vec(cores.size, UInt(64.W)))
  })

  val composition = Instantiate(new Composition(
    memory,
    placements,
    regions,
    tracking,
    workers,
    workers.map(topology.signatures)
  ))

  // A CPU-only core 0 issues bank moves only into the shared bank mesh.
  val controllerMove = shared && cores.head.buckyball.isEmpty

  val banks = Option.when(shared)(
    Instantiate(new BankNetwork(base.get, enabled, useMesh = true, controllerMove = controllerMove))
  )

  composition.io.resetVector := io.resetVector
  composition.io.interrupts  := io.interrupts
  io.uncachedRequest <> composition.io.uncachedRequest
  composition.io.uncachedResponse <> io.uncachedResponse
  io.backingRequest <> composition.io.backingReq
  composition.io.backingResp <> io.backingResponse
  io.retired                 := composition.io.retired
  io.retiredPc               := composition.io.retiredPc
  io.trapped                 := composition.io.trapped
  io.trapCause               := composition.io.trapCause
  io.trapValue               := composition.io.trapValue
  io.trapPc                  := composition.io.trapPc
  banks.foreach(_.io.hartIds := VecInit(topology.hartIds.map(_.U(64.W))))
  io.failure                 := 0.U.asTypeOf(io.failure)

  for (i <- cores.indices) {
    cores(i).buckyball match {
      case Some(b) =>
        val accelerator = Instantiate(new BuckyballAccelerator(b))
        val admission   = Instantiate(new Admission(b, tracking, prepared, regions, axiParams)(cpuParameters(i)))
        admission.io.core <> composition.io.admission(i)
        admission.io.task <> composition.io.taskControl(i)
        if (i == 0) composition.io.controllerSatp := admission.io.taskSatp
        accelerator.io.cmd <> admission.io.npuCommand
        admission.io.npuResponse <> accelerator.io.resp
        admission.io.allocation.valid             := accelerator.io.allocation.valid
        admission.io.allocation.bits              := accelerator.io.allocation.bits.rob_id
        admission.io.retired                      := accelerator.io.retired
        admission.io.npuFault                     := accelerator.io.fault
        admission.io.npuBusy                      := !accelerator.io.idle
        admission.io.npuInterrupt                 := accelerator.io.interrupt
        admission.io.footprints                   := accelerator.io.footprints
        admission.io.dma <> accelerator.io.dma
        val endpoint = enabled.indexOf(i)
        io.dma(endpoint) <> admission.io.axi
        accelerator.io.hartid                := topology.hartIds(i).U
        accelerator.io.sharedBankOwnerHartId := topology.hartIds(topology.controller.getOrElse(i)).U
        banks match {
          case Some(network) =>
            val port = network.io.compute(endpoint)
            port.requests <> accelerator.io.shared_mem_req
            port.move <> accelerator.io.mvover
            accelerator.io.meshLocalBank <> port.local
            port.config <> accelerator.io.shared_config
            port.queryValid                             := accelerator.io.shared_query_valid
            port.queryVbank                             := accelerator.io.shared_query_vbank_id
            accelerator.io.shared_query_group_count     := port.queryGroups
            port.barrierArrive                          := accelerator.io.barrier_arrive
            accelerator.io.barrier_release              := port.barrierRelease
            accelerator.io.shared_bank_hashes.foreach(_ := network.io.bankHashes.get)
          case None          =>
            // Private banks: no shared channels exist, and mesh moves and barriers stay idle.
            accelerator.io.mvover.command.ready         := false.B
            accelerator.io.mvover.completion.valid      := false.B
            accelerator.io.mvover.completion.bits       := false.B
            accelerator.io.meshLocalBank.request.valid  := false.B
            accelerator.io.meshLocalBank.request.bits   := 0.U.asTypeOf(accelerator.io.meshLocalBank.request.bits)
            accelerator.io.meshLocalBank.response.ready := true.B
            accelerator.io.shared_config.ready          := false.B
            accelerator.io.shared_query_group_count     := 0.U
            accelerator.io.barrier_release              := false.B
        }
        io.failure(i).valid                  := admission.io.halted
        io.failure(i).bits                   := admission.io.fault
        io.workDrained(i)                    := admission.io.workDrained
      case None    =>
        val admission =
          Instantiate(new ControllerAdmission(tracking, c, moves = i == 0 && controllerMove)(cpuParameters(i)))
        admission.io.core <> composition.io.admission(i)
        admission.io.task <> composition.io.taskControl(i)
        if (i == 0) composition.io.controllerSatp := admission.io.taskSatp
        if (i == 0 && controllerMove) {
          banks.get.io.controllerMvover.get <> admission.io.move
        } else {
          admission.io.move.command.ready    := false.B
          admission.io.move.completion.valid := false.B
          admission.io.move.completion.bits  := false.B
        }
        io.failure(i)                             := admission.io.moveFault
        io.workDrained(i)                         := composition.io.admission(i).outstanding === 0.U
    }
  }
}
