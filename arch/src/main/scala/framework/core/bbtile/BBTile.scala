package framework.core.bbtile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.lazymodule._

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.devices.tilelink.{BasicBusBlocker, BasicBusBlockerParams}
import freechips.rocketchip.diplomacy.{AddressSet, BufferParams, DisableMonitors}
import freechips.rocketchip.resources.{
  Description,
  Resource,
  ResourceAddress,
  ResourceAnchors,
  ResourceBinding,
  ResourceBindings,
  SimpleDevice
}
import freechips.rocketchip.interrupts.IntIdentityNode
import freechips.rocketchip.tilelink.{
  TLBuffer,
  TLClientNode,
  TLClientParameters,
  TLIdentityNode,
  TLMasterPortParameters,
  TLWidthWidget,
  TLXbar
}
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.prci.{ClockCrossingType, ClockSinkParameters, RationalCrossing}
import freechips.rocketchip.util.{Annotated, InOrderArbiter}
import freechips.rocketchip.util.BooleanToAugmentedBoolean

import framework.top.GlobalConfig
import framework.core.bbtile.id.RVVRoCCDecode
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.shared.SharedMemBackend
import framework.memdomain.frontend.outside_channel.MemConfigerIO

/**
 * BBTile — a composable tile containing Rocket core(s) + optional Buckyball accelerator(s).
 *
 * When nCores=1 (default), behaviour is identical to the original single-core tile.
 * When nCores>1, the tile contains N (RocketBB + BuckyballAccelerator) pairs that share
 * a single SharedMemBackend and BarrierUnit.
 *
 * The trait-provided DCache/ICache/PTW serve core-0.  Cores 1..N-1 are wired entirely
 * inside BBTileModuleImp (no extra diplomacy DCache/ICache — they share core-0's cache
 * hierarchy via the same TL xbar for now; independent caches are a future enhancement).
 */
class BBTile private (
  val bbParams: BBTileParams,
  crossing:     ClockCrossingType,
  lookup:       LookupByHartIdImpl,
  q:            Parameters)
    extends BaseTile(bbParams, crossing, lookup, q)
    with SinksExternalInterrupts
    with SourcesExternalNotifications
    with HasHellaCache
    with HasICacheFrontend {

  def this(
    params:     BBTileParams,
    crossing:   HierarchicalElementCrossingParamsLike,
    lookup:     LookupByHartIdImpl
  )(
    implicit p: Parameters
  ) =
    this(params, crossing.crossingType, lookup, BBTile.injectBuildRoCC(p, params.withBuckyball))

  val nCores = bbParams.nCores

  // RoCC CSRs — Buckyball doesn't use custom CSRs, so this is always empty
  val roccCSRs: Seq[Seq[CustomCSR]] = Nil

  // ---------------------------------------------------------------------------
  // Diplomacy nodes — tile boundary
  // ---------------------------------------------------------------------------
  val intOutwardNode = bbParams.beuAddr.map(_ => IntIdentityNode())
  val slaveNode      = TLIdentityNode()
  val masterNode     = visibilityNode

  // Scratchpad (DTIM)
  val dtim_adapter = bbParams.dcache.flatMap { d =>
    d.scratch.map { s =>
      LazyModule(new ScratchpadSlavePort(
        AddressSet.misaligned(s, d.dataScratchpadBytes),
        lazyCoreParamsView.coreDataBytes,
        bbParams.core.useAtomics && !bbParams.core.useAtomicsOnlyForIO
      ))
    }
  }

  dtim_adapter.foreach(lm => connectTLSlave(lm.node, lm.node.portParams.head.beatBytes))

  // Bus error unit
  val bus_error_unit = bbParams.beuAddr.map { a =>
    val beu = LazyModule(new BusErrorUnit(new L1BusErrors, BusErrorUnitParams(a), xLen / 8))
    intOutwardNode.get := beu.intNode
    connectTLSlave(beu.node, xBytes)
    beu
  }

  // Master port blocker
  val tile_master_blocker =
    bbParams.blockerCtrlAddr
      .map(BasicBusBlockerParams(_, xBytes, masterPortBeatBytes, deadlock = true))
      .map(bp => LazyModule(new BasicBusBlocker(bp)))

  tile_master_blocker.foreach(lm => connectTLSlave(lm.controlNode, xBytes))

  // ---------------------------------------------------------------------------
  // Buckyball accelerator TileLink nodes (diplomacy layer) — N pairs of DMA
  // ---------------------------------------------------------------------------
  val bbConfig = bbParams.buckyballConfig

  val bb_reader_nodes: Seq[Option[TLClientNode]] = (0 until nCores).map { i =>
    if (bbParams.withBuckyball) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = s"bb-dma-reader-$i",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbConfig.memDomain.dma_n_xacts)
    ))))))
    else None
  }

  val bb_writer_nodes: Seq[Option[TLClientNode]] = (0 until nCores).map { i =>
    if (bbParams.withBuckyball) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = s"bb-dma-writer-$i",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbConfig.memDomain.dma_n_xacts)
    ))))))
    else None
  }

  // Gather all DMA nodes into one xbar
  if (bbParams.withBuckyball) {
    val bb_xbar = TLXbar()
    for (i <- 0 until nCores) {
      bb_xbar := TLBuffer() := bb_reader_nodes(i).get
      bb_xbar := TLBuffer() := bb_writer_nodes(i).get
    }
    tlOtherMastersNode :=* TLWidthWidget(bbConfig.memDomain.dma_buswidth / 8) := TLBuffer() := bb_xbar
  }

  // ---------------------------------------------------------------------------
  // TileLink topology
  // ---------------------------------------------------------------------------
  tlOtherMastersNode := tile_master_blocker.map(_.node := tlMasterXbar.node).getOrElse(tlMasterXbar.node)
  masterNode :=* tlOtherMastersNode
  DisableMonitors(implicit p => tlSlaveXbar.node :*= slaveNode)

  // DCache port count: core + PTW(via usingVM) + DTIM + vector + RoCC tieoff
  nDCachePorts += 1 + (dtim_adapter.isDefined).toInt +
    bbParams.core.vector.map(_.useDCache.toInt).getOrElse(0) +
    bbParams.withBuckyball.toInt

  // ---------------------------------------------------------------------------
  // Device tree properties
  // ---------------------------------------------------------------------------
  val dtimProperty = dtim_adapter.map(d => Map("sifive,dtim" -> d.device.asProperty)).getOrElse(Nil)
  val itimProperty = frontend.icache.itimProperty.toSeq.flatMap(p => Map("sifive,itim" -> p))
  val beuProperty  = bus_error_unit.map(d => Map("sifive,buserror" -> d.device.asProperty)).getOrElse(Nil)

  val cpuDevice: SimpleDevice = new SimpleDevice("cpu", Seq("sifive,rocket0", "riscv")) {
    override def parent = Some(ResourceAnchors.cpus)

    override def describe(resources: ResourceBindings): Description = {
      val Description(name, mapping) = super.describe(resources)
      Description(
        name,
        mapping ++ cpuProperties ++ nextLevelCacheProperty
          ++ tileProperties ++ dtimProperty ++ itimProperty ++ beuProperty
      )
    }

  }

  // Vector unit (optional)
  val vector_unit = bbParams.core.vector.map(v => LazyModule(v.build(p)))
  vector_unit.foreach(vu => tlMasterXbar.node :=* vu.atlNode)
  vector_unit.foreach(vu => tlOtherMastersNode :=* vu.tlNode)

  ResourceBinding {
    Resource(cpuDevice, "reg").bind(ResourceAddress(bbParams.tileId))
  }

  // Buckyball needs one PTW port per accelerator
  if (bbParams.withBuckyball) {
    nPTWPorts += nCores
  }

  override lazy val module = new BBTileModuleImp(this)

  override def makeMasterBoundaryBuffers(crossing: ClockCrossingType)(implicit p: Parameters) =
    (bbParams.boundaryBuffers, crossing) match {
      case (Some(RocketTileBoundaryBufferParams(true)), _) => TLBuffer()
      case (Some(RocketTileBoundaryBufferParams(false)), _: RationalCrossing) =>
        TLBuffer(BufferParams.none, BufferParams.flow, BufferParams.none, BufferParams.flow, BufferParams(1))
      case _                                               => TLBuffer(BufferParams.none)
    }

  override def makeSlaveBoundaryBuffers(crossing: ClockCrossingType)(implicit p: Parameters) =
    (bbParams.boundaryBuffers, crossing) match {
      case (Some(RocketTileBoundaryBufferParams(true)), _) => TLBuffer()
      case (Some(RocketTileBoundaryBufferParams(false)), _: RationalCrossing) =>
        TLBuffer(BufferParams.flow, BufferParams.none, BufferParams.none, BufferParams.none, BufferParams.none)
      case _                                               => TLBuffer(BufferParams.none)
    }

}

// =============================================================================
// Module implementation (Chisel layer)
// =============================================================================
class BBTileModuleImp(outer: BBTile) extends BaseTileModuleImp(outer) with HasICacheFrontendModule {

  Annotated.params(this, outer.bbParams)
  val nCores = outer.nCores

  // --- FPU (optional) ---
  val fpuOpt = outer.bbParams.core.fpu.map(params => Module(new FPU(params)(outer.p)))

  // --- Rocket core (using our fork that accepts BBTile) ---
  val core = Module(new RocketBB(outer)(outer.p))

  // Vector unit connections
  outer.vector_unit.foreach { v =>
    core.io.vector.get <> v.module.io.core
    v.module.io.tlb <> outer.dcache.module.io.tlb_port
  }

  core.io.reset_vector := DontCare

  // Report conditions
  outer.reportHalt(List(outer.dcache.module.io.errors))
  outer.reportCease(outer.bbParams.core.clockGate.option(
    !outer.dcache.module.io.cpu.clock_enabled &&
      !outer.frontend.module.io.cpu.clock_enabled &&
      !ptw.io.dpath.clock_enabled &&
      core.io.cease
  ))
  outer.reportWFI(Some(core.io.wfi))

  // Interrupts
  outer.decodeCoreInterrupts(core.io.interrupts)
  outer.bus_error_unit.foreach { beu =>
    core.io.interrupts.buserror.get := beu.module.io.interrupt
    beu.module.io.errors.dcache     := outer.dcache.module.io.errors
    beu.module.io.errors.icache     := outer.frontend.module.io.errors
  }
  core.io.interrupts.nmi.foreach(nmi => nmi := outer.nmiSinkNode.get.bundle)

  // Trace and misc
  outer.traceSourceNode.bundle <> core.io.trace
  core.io.traceStall := outer.traceAuxSinkNode.bundle.stall
  outer.bpwatchSourceNode.bundle <> core.io.bpwatch
  core.io.hartid     := outer.hartIdSinkNode.bundle

  // Core pipeline connections
  outer.frontend.module.io.cpu <> core.io.imem
  dcachePorts += core.io.dmem

  // FPU
  fpuOpt.foreach { fpu =>
    core.io.fpu :<>= fpu.io.waiveAs[FPUCoreIO](_.cp_req, _.cp_resp)
    fpu.io.cp_req.valid  := false.B
    fpu.io.cp_req.bits   := DontCare
    fpu.io.cp_resp.ready := false.B
  }
  if (fpuOpt.isEmpty) {
    core.io.fpu := DontCare
  }

  // Vector unit DCache port
  outer.vector_unit.foreach { v =>
    if (outer.bbParams.core.vector.get.useDCache) {
      dcachePorts += v.module.io.dmem
    } else {
      v.module.io.dmem := DontCare
    }
  }

  core.io.ptw <> ptw.io.dpath

  // DTIM adapter
  outer.dtim_adapter.foreach(lm => dcachePorts += lm.module.io.dmem)

  // ---------------------------------------------------------------------------
  // Helper: wire a BuckyballAccelerator's PTW to tile's PTW subsystem
  // ---------------------------------------------------------------------------
  def wireBBPtw(buckyball: BuckyballAccelerator): Unit = {
    val bbPtw = Wire(new TLBPTWIO)
    ptwPorts += bbPtw
    bbPtw.req.valid               := buckyball.io.ptw(0).req.valid
    bbPtw.req.bits.valid          := buckyball.io.ptw(0).req.bits.valid
    bbPtw.req.bits.bits.addr      := buckyball.io.ptw(0).req.bits.bits.addr
    bbPtw.req.bits.bits.need_gpa  := buckyball.io.ptw(0).req.bits.bits.need_gpa
    bbPtw.req.bits.bits.vstage1   := buckyball.io.ptw(0).req.bits.bits.vstage1
    bbPtw.req.bits.bits.stage2    := buckyball.io.ptw(0).req.bits.bits.stage2
    buckyball.io.ptw(0).req.ready := bbPtw.req.ready

    buckyball.io.ptw(0).resp.valid                          := bbPtw.resp.valid
    buckyball.io.ptw(0).resp.bits.ae_ptw                    := bbPtw.resp.bits.ae_ptw
    buckyball.io.ptw(0).resp.bits.ae_final                  := bbPtw.resp.bits.ae_final
    buckyball.io.ptw(0).resp.bits.pf                        := bbPtw.resp.bits.pf
    buckyball.io.ptw(0).resp.bits.gf                        := bbPtw.resp.bits.gf
    buckyball.io.ptw(0).resp.bits.hr                        := bbPtw.resp.bits.hr
    buckyball.io.ptw(0).resp.bits.hw                        := bbPtw.resp.bits.hw
    buckyball.io.ptw(0).resp.bits.hx                        := bbPtw.resp.bits.hx
    buckyball.io.ptw(0).resp.bits.pte.ppn                   := bbPtw.resp.bits.pte.ppn
    buckyball.io.ptw(0).resp.bits.pte.reserved_for_future   := bbPtw.resp.bits.pte.reserved_for_future
    buckyball.io.ptw(0).resp.bits.pte.reserved_for_software := bbPtw.resp.bits.pte.reserved_for_software
    buckyball.io.ptw(0).resp.bits.pte.d                     := bbPtw.resp.bits.pte.d
    buckyball.io.ptw(0).resp.bits.pte.a                     := bbPtw.resp.bits.pte.a
    buckyball.io.ptw(0).resp.bits.pte.g                     := bbPtw.resp.bits.pte.g
    buckyball.io.ptw(0).resp.bits.pte.u                     := bbPtw.resp.bits.pte.u
    buckyball.io.ptw(0).resp.bits.pte.x                     := bbPtw.resp.bits.pte.x
    buckyball.io.ptw(0).resp.bits.pte.w                     := bbPtw.resp.bits.pte.w
    buckyball.io.ptw(0).resp.bits.pte.r                     := bbPtw.resp.bits.pte.r
    buckyball.io.ptw(0).resp.bits.pte.v                     := bbPtw.resp.bits.pte.v
    buckyball.io.ptw(0).resp.bits.level                     := bbPtw.resp.bits.level
    buckyball.io.ptw(0).resp.bits.fragmented_superpage      := bbPtw.resp.bits.fragmented_superpage
    buckyball.io.ptw(0).resp.bits.homogeneous               := bbPtw.resp.bits.homogeneous
    buckyball.io.ptw(0).resp.bits.gpa.valid                 := bbPtw.resp.bits.gpa.valid
    buckyball.io.ptw(0).resp.bits.gpa.bits                  := bbPtw.resp.bits.gpa.bits
    buckyball.io.ptw(0).resp.bits.gpa_is_pte                := bbPtw.resp.bits.gpa_is_pte

    buckyball.io.ptw(0).ptbr.mode  := bbPtw.ptbr.mode
    buckyball.io.ptw(0).ptbr.asid  := bbPtw.ptbr.asid
    buckyball.io.ptw(0).ptbr.ppn   := bbPtw.ptbr.ppn
    buckyball.io.ptw(0).hgatp.mode := bbPtw.hgatp.mode
    buckyball.io.ptw(0).hgatp.asid := bbPtw.hgatp.asid
    buckyball.io.ptw(0).hgatp.ppn  := bbPtw.hgatp.ppn
    buckyball.io.ptw(0).vsatp.mode := bbPtw.vsatp.mode
    buckyball.io.ptw(0).vsatp.asid := bbPtw.vsatp.asid
    buckyball.io.ptw(0).vsatp.ppn  := bbPtw.vsatp.ppn
    buckyball.io.ptw(0).status     := bbPtw.status
    buckyball.io.ptw(0).hstatus    := bbPtw.hstatus
    buckyball.io.ptw(0).gstatus    := bbPtw.gstatus
    buckyball.io.ptw(0).pmp.zipWithIndex.foreach { case (pmpPort, i) =>
      pmpPort.cfg.l   := bbPtw.pmp(i).cfg.l
      pmpPort.cfg.res := bbPtw.pmp(i).cfg.res
      pmpPort.cfg.a   := bbPtw.pmp(i).cfg.a
      pmpPort.cfg.x   := bbPtw.pmp(i).cfg.x
      pmpPort.cfg.w   := bbPtw.pmp(i).cfg.w
      pmpPort.cfg.r   := bbPtw.pmp(i).cfg.r
      pmpPort.addr    := bbPtw.pmp(i).addr
      pmpPort.mask    := bbPtw.pmp(i).mask
    }
    buckyball.io.ptw(0).customCSRs := DontCare
    bbPtw.customCSRs               := DontCare
  }

  // ---------------------------------------------------------------------------
  // Buckyball accelerators — N instances sharing SharedMemBackend + BarrierUnit
  // ---------------------------------------------------------------------------
  if (outer.bbParams.withBuckyball) {
    val bankChannel = outer.bbConfig.memDomain.bankChannel

    // Instantiate N accelerators
    val accelerators = (0 until nCores).map { i =>
      val (tl_reader, edge) = outer.bb_reader_nodes(i).get.out(0)
      val (tl_writer, _)    = outer.bb_writer_nodes(i).get.out(0)
      val acc               = Module(new BuckyballAccelerator(outer.bbConfig)(edge))
      acc.io.hartid := outer.hartIdSinkNode.bundle + i.U

      // DMA TileLink
      tl_reader <> acc.io.tl_reader
      tl_writer <> acc.io.tl_writer

      // PTW
      wireBBPtw(acc)

      // TLB exception
      acc.io.tlbExp(0).flush_skip  := false.B
      acc.io.tlbExp(0).flush_retry := false.B

      // CPU sfence → Buckyball TLB flush
      acc.io.sfence := ptw.io.dpath.sfence.valid

      acc
    }

    // Core-0 RoCC wiring (the single Rocket core drives accelerator 0)
    accelerators(0).io.cmd <> core.io.rocc.cmd
    core.io.rocc.resp <> accelerators(0).io.resp
    core.io.rocc.busy      := accelerators(0).io.busy
    core.io.rocc.interrupt := accelerators(0).io.interrupt

    // RoCC mem: tied-off HellaCacheIF for the DCache arbiter port count
    val roccMemIF = Module(new SimpleHellaCacheIF())
    roccMemIF.io.requestor.req.valid          := false.B
    roccMemIF.io.requestor.req.bits           := DontCare
    roccMemIF.io.requestor.s1_kill            := false.B
    roccMemIF.io.requestor.s1_data            := DontCare
    roccMemIF.io.requestor.s2_kill            := false.B
    roccMemIF.io.requestor.keep_clock_enabled := false.B
    dcachePorts += roccMemIF.io.cache
    core.io.rocc.mem                          := DontCare

    // SharedMemBackend (tile-level singleton)
    val sharedBackend = Module(new SharedMemBackend(outer.bbConfig))

    // Connect each accelerator's shared ports to the SharedMemBackend
    for (i <- 0 until nCores) {
      for (ch <- 0 until bankChannel) {
        val slot = i * bankChannel + ch
        sharedBackend.io.mem_req(slot) <> accelerators(i).io.shared_mem_req(ch)
      }
      // Shared query: connect per-core query to shared backend
      // (only one query port on SharedMemBackend — for now use accelerator 0's query;
      //  each accelerator's MemBackend routes shared queries through its IO)
    }

    // Shared config arbiter: N accelerators → 1 SharedMemBackend config port
    val cfgArb = Module(new Arbiter(new MemConfigerIO(outer.bbConfig), nCores))
    for (i <- 0 until nCores) {
      cfgArb.io.in(i) <> accelerators(i).io.shared_config
    }
    sharedBackend.io.config <> cfgArb.io.out

    // Shared query — simplified: each accelerator queries independently,
    // but SharedMemBackend has one query port. Use accelerator 0's for now.
    sharedBackend.io.query_vbank_id             := accelerators(0).io.shared_query_vbank_id
    accelerators(0).io.shared_query_group_count := sharedBackend.io.query_group_count
    for (i <- 1 until nCores) {
      accelerators(i).io.shared_query_group_count := sharedBackend.io.query_group_count
    }

    // BarrierUnit (tile-level singleton)
    val barrierUnit = Module(new BarrierUnit(nCores))
    for (i <- 0 until nCores) {
      barrierUnit.io.arrive(i)           := accelerators(i).io.barrier_arrive
      accelerators(i).io.barrier_release := barrierUnit.io.release(i)
    }

  } else {
    // No accelerator — tie off RoCC
    core.io.rocc.cmd.ready  := false.B
    core.io.rocc.resp.valid := false.B
    core.io.rocc.resp.bits  := DontCare
    core.io.rocc.busy       := DontCare
    core.io.rocc.interrupt  := DontCare
    core.io.rocc.mem        := DontCare
  }

  // --- Finalize DCache arbiter and PTW connections (after all ports added) ---
  val h = dcachePorts.size
  val c = core.dcacheArbPorts
  val o = outer.nDCachePorts
  require(h == c, s"port list size was $h, core expected $c")
  require(h == o, s"port list size was $h, outer counted $o")

  dcacheArb.io.requestor <> dcachePorts.toSeq
  ptw.io.requestor <> ptwPorts.toSeq
}

object BBTile {

  /**
   * Inject a dummy BuildRoCC entry so that usingRoCC=true throughout all
   * HasRocketCoreParameters mixins (CSR, decode, etc.), without actually
   * using the LazyRoCC mechanism.
   */
  def injectBuildRoCC(p: Parameters, withBuckyball: Boolean): Parameters =
    if (withBuckyball)
      p.alterPartial { case BuildRoCC => Seq((_: Parameters) => null.asInstanceOf[LazyRoCC]) }
    else p

}
