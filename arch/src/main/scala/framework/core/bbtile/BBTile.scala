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
  TLCacheCork,
  TLClientNode,
  TLClientParameters,
  TLFragmenter,
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
import sifive.blocks.inclusivecache.{CacheParameters, InclusiveCache, InclusiveCacheMicroParameters}

/**
 * BBTile — a composable tile containing one Rocket core + optional per-core-index Buckyball slots.
 *
 * nCores controls buckyball slot count and shared backend sizing.
 * Rocket is still a single core today; RoCC is wired to slot-0.
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
    this(params, crossing.crossingType, lookup, BBTile.injectBuildRoCC(p, params.withAnyBuckyball, params.nCores))

  val nCores           = bbParams.nCores
  val bbPerCore        = bbParams.resolvedBuckyballPerCore
  require(bbPerCore.size == nCores, s"bbPerCore size (${bbPerCore.size}) must equal nCores ($nCores)")
  val bbEnabledCoreIds = bbPerCore.zipWithIndex.collect { case (Some(_), i) => i }
  val hasBuckyball     = bbEnabledCoreIds.nonEmpty
  val bbSharedConfig   = bbPerCore.collectFirst { case Some(cfg) => cfg }
  if (hasBuckyball) {
    require(
      bbEnabledCoreIds.contains(0),
      "core-0 must enable buckyball because RoCC cmd/resp is currently wired to core-0 only"
    )
    val cfg0 = bbSharedConfig.get
    require(cfg0.top.nCores == nCores, s"buckyball top.nCores (${cfg0.top.nCores}) must equal tile nCores ($nCores)")
    bbEnabledCoreIds.foreach { i =>
      val cfg = bbPerCore(i).get
      require(
        cfg.top.nCores == nCores,
        s"core-$i buckyball top.nCores (${cfg.top.nCores}) must equal tile nCores ($nCores)"
      )
      require(cfg.memDomain.bankChannel == cfg0.memDomain.bankChannel, s"core-$i bankChannel mismatch")
      require(cfg.memDomain.dma_buswidth == cfg0.memDomain.dma_buswidth, s"core-$i dma_buswidth mismatch")
    }
  }

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
  val extraFrontends: Seq[Frontend] = (1 until nCores).map { _ =>
    val f = LazyModule(new Frontend(bbParams.icache.get, bbParams.tileId))
    tlMasterXbar.node                               := TLWidthWidget(bbParams.icache.get.rowBits / 8) := f.masterNode
    connectTLSlave(f.slaveNode, bbParams.core.fetchBytes)
    f.icache.hartIdSinkNodeOpt.foreach(_            := hartIdNexusNode)
    f.icache.mmioAddressPrefixSinkNodeOpt.foreach(_ := mmioAddressPrefixNexusNode)
    f.resetVectorSinkNode                           := resetVectorNexusNode
    f
  }

  nPTWPorts += extraFrontends.size

  val bb_reader_nodes: Seq[Option[TLClientNode]] = (0 until nCores).map { i =>
    if (bbPerCore(i).isDefined) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = s"bb-dma-reader-$i",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbPerCore(i).get.memDomain.dma_n_xacts)
    ))))))
    else None
  }

  val bb_writer_nodes: Seq[Option[TLClientNode]] = (0 until nCores).map { i =>
    if (bbPerCore(i).isDefined) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = s"bb-dma-writer-$i",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbPerCore(i).get.memDomain.dma_n_xacts)
    ))))))
    else None
  }

  // Gather all DMA nodes into one xbar
  if (hasBuckyball) {
    val cfg0    = bbSharedConfig.get
    val bb_xbar = TLXbar()
    for (i <- bbEnabledCoreIds) {
      bb_xbar := TLBuffer() := bb_reader_nodes(i).get
      bb_xbar := TLBuffer() := bb_writer_nodes(i).get
    }
    tlOtherMastersNode :=* TLWidthWidget(cfg0.memDomain.dma_buswidth / 8) := TLBuffer() := bb_xbar
  }

  // ---------------------------------------------------------------------------
  // Per-tile private L2 cache (optional)
  // ---------------------------------------------------------------------------
  val tileL2 = bbParams.l2cache.map { l2params =>
    val l2 = LazyModule(new InclusiveCache(
      CacheParameters(
        level = 2,
        ways = l2params.ways,
        sets = l2params.sets,
        blockBytes = p(freechips.rocketchip.subsystem.CacheBlockBytes),
        beatBytes = masterPortBeatBytes,
        hintsSkipProbe = false
      ),
      InclusiveCacheMicroParameters(
        writeBytes = l2params.writeBytes,
        portFactor = l2params.portFactor,
        memCycles = l2params.memCycles,
        innerBuf = l2params.bufInnerInterior,
        outerBuf = l2params.bufOuterInterior
      ),
      None // No control port for per-tile L2
    ))
    l2.suggestName(s"tile_l2_${bbParams.tileId}")

    // Create buffers and cork for L2
    val l2_inner_buffer = l2params.bufInnerExterior()
    val l2_outer_buffer = l2params.bufOuterExterior()
    val cork            = LazyModule(new TLCacheCork)

    l2_inner_buffer.suggestName(s"tile_l2_${bbParams.tileId}_inner_buffer")
    l2_outer_buffer.suggestName(s"tile_l2_${bbParams.tileId}_outer_buffer")
    cork.suggestName(s"tile_l2_${bbParams.tileId}_cork")

    (l2, l2_inner_buffer, l2_outer_buffer, cork)
  }

  // ---------------------------------------------------------------------------
  // TileLink topology
  // ---------------------------------------------------------------------------
  // SCU moved to subsystem level (CBUS) as a global multi-hart device.
  // See sims/scu/SCU.scala and WithSCU config.

  tlOtherMastersNode := tile_master_blocker.map(_.node := tlMasterXbar.node).getOrElse(tlMasterXbar.node)

  // Route through L2 if present, otherwise connect directly to masterNode
  tileL2 match {
    case Some((l2, innerBuf, outerBuf, cork)) =>
      // Topology: tlOtherMastersNode -> innerBuf -> L2 -> outerBuf -> cork -> masterNode
      innerBuf.node :*= tlOtherMastersNode
      l2.node :*= innerBuf.node
      outerBuf.node :*= l2.node
      cork.node :*= outerBuf.node
      masterNode :=* cork.node
    case None                                 =>
      // Direct connection (original behavior)
      masterNode :=* tlOtherMastersNode
  }

  DisableMonitors(implicit p => tlSlaveXbar.node :*= slaveNode)

  // DCache port count: core + PTW(via usingVM) + DTIM + vector + RoCC tieoff
  nDCachePorts += nCores + (dtim_adapter.isDefined).toInt +
    bbParams.core.vector.map(_.useDCache.toInt).getOrElse(0) +
    hasBuckyball.toInt

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
  if (hasBuckyball) {
    nPTWPorts += bbEnabledCoreIds.size
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
  val fpuOpts = Seq.fill(nCores)(outer.bbParams.core.fpu.map(params => Module(new FPU(params)(outer.p))))

  // --- Rocket core (using our fork that accepts BBTile) ---
  val cores = (0 until nCores).map { i =>
    Module(new RocketBB(outer, outer.bbPerCore(i).isDefined)(outer.p))
  }

  val core = cores.head

  // Vector unit connections
  outer.vector_unit.foreach { v =>
    core.io.vector.get <> v.module.io.core
    v.module.io.tlb <> outer.dcache.module.io.tlb_port
  }

  core.io.reset_vector                 := DontCare
  cores.tail.foreach(_.io.reset_vector := DontCare)

  // Report conditions
  outer.reportHalt(List(outer.dcache.module.io.errors))
  outer.reportCease(outer.bbParams.core.clockGate.option(
    !outer.dcache.module.io.cpu.clock_enabled &&
      !outer.frontend.module.io.cpu.clock_enabled &&
      !ptw.io.dpath.clock_enabled &&
      cores.map(_.io.cease).reduce(_ && _)
  ))
  outer.reportWFI(Some(cores.map(_.io.wfi).reduce(_ && _)))

  // Interrupts
  cores.foreach(c => outer.decodeCoreInterrupts(c.io.interrupts))
  outer.bus_error_unit.foreach { beu =>
    cores.foreach(_.io.interrupts.buserror.get := beu.module.io.interrupt)
    beu.module.io.errors.dcache                := outer.dcache.module.io.errors
    beu.module.io.errors.icache                := outer.frontend.module.io.errors
  }
  cores.foreach(_.io.interrupts.nmi.foreach(nmi => nmi := outer.nmiSinkNode.get.bundle))

  // Trace and misc
  outer.traceSourceNode.bundle <> core.io.trace
  cores.foreach(_.io.traceStall := outer.traceAuxSinkNode.bundle.stall)
  outer.bpwatchSourceNode.bundle <> core.io.bpwatch
  cores.zipWithIndex.foreach { case (c, i) => c.io.hartid := outer.hartIdSinkNode.bundle * nCores.U + i.U }

  // Core pipeline connections
  outer.frontend.module.io.cpu <> core.io.imem
  for ((f, i) <- outer.extraFrontends.zipWithIndex) {
    f.module.io.cpu <> cores(i + 1).io.imem
    ptwPorts += f.module.io.ptw
  }
  cores.foreach(c => dcachePorts += c.io.dmem)

  // FPU
  for ((c, fpuOpt) <- cores.zip(fpuOpts)) {
    fpuOpt.foreach { fpu =>
      c.io.fpu :<>= fpu.io.waiveAs[FPUCoreIO](_.cp_req, _.cp_resp)
      fpu.io.cp_req.valid  := false.B
      fpu.io.cp_req.bits   := DontCare
      fpu.io.cp_resp.ready := false.B
    }
    if (fpuOpt.isEmpty) {
      c.io.fpu := DontCare
    }
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
  cores.tail.foreach(_.io.ptw := DontCare)

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
  def tieOffMemReq(req: MemRequestIO): Unit = {
    req.write.req.valid  := false.B
    req.write.req.bits   := DontCare
    req.write.resp.ready := true.B
    req.read.req.valid   := false.B
    req.read.req.bits    := DontCare
    req.read.resp.ready  := true.B
    req.bank_id          := 0.U
    req.group_id         := 0.U
    req.is_shared        := false.B
    req.hart_id          := 0.U
  }

  if (outer.hasBuckyball) {
    val cfg0        = outer.bbSharedConfig.get
    val bankChannel = cfg0.memDomain.bankChannel

    // Instantiate accelerators for enabled cores only
    val accelerators = (0 until nCores).map { i =>
      outer.bbPerCore(i).map { cfg =>
        val (tl_reader, edge) = outer.bb_reader_nodes(i).get.out(0)
        val (tl_writer, _)    = outer.bb_writer_nodes(i).get.out(0)
        val acc               = Module(new BuckyballAccelerator(cfg)(edge))
        acc.io.hartid := outer.hartIdSinkNode.bundle * nCores.U + i.U

        tl_reader <> acc.io.tl_reader
        tl_writer <> acc.io.tl_writer
        wireBBPtw(acc)
        acc.io.tlbExp(0).flush_skip  := false.B
        acc.io.tlbExp(0).flush_retry := false.B
        acc.io.sfence                := ptw.io.dpath.sfence.valid
        acc
      }
    }

    val enabledAccelerators = outer.bbEnabledCoreIds.map(i => accelerators(i).get)
    for (i <- 0 until nCores) {
      accelerators(i) match {
        case Some(acc) =>
          acc.io.cmd <> cores(i).io.rocc.cmd
          cores(i).io.rocc.resp <> acc.io.resp
          cores(i).io.rocc.busy      := acc.io.busy
          cores(i).io.rocc.interrupt := acc.io.interrupt
        case None      =>
          cores(i).io.rocc.cmd.ready  := false.B
          cores(i).io.rocc.resp.valid := false.B
          cores(i).io.rocc.resp.bits  := DontCare
          cores(i).io.rocc.busy       := false.B
          cores(i).io.rocc.interrupt  := false.B
      }
    }

    // RoCC mem: tied-off HellaCacheIF for the DCache arbiter port count
    val roccMemIF = Module(new SimpleHellaCacheIF())
    roccMemIF.io.requestor.req.valid          := false.B
    roccMemIF.io.requestor.req.bits           := DontCare
    roccMemIF.io.requestor.s1_kill            := false.B
    roccMemIF.io.requestor.s1_data            := DontCare
    roccMemIF.io.requestor.s2_kill            := false.B
    roccMemIF.io.requestor.keep_clock_enabled := false.B
    dcachePorts += roccMemIF.io.cache
    cores.foreach(_.io.rocc.mem               := DontCare)

    // SharedMemBackend (tile-level singleton)
    val sharedBackend = Module(new SharedMemBackend(cfg0))

    // Connect each accelerator's shared ports to the SharedMemBackend
    for (i <- 0 until nCores) {
      for (ch <- 0 until bankChannel) {
        val slot = i * bankChannel + ch
        accelerators(i) match {
          case Some(acc) => sharedBackend.io.mem_req(slot) <> acc.io.shared_mem_req(ch)
          case None      => tieOffMemReq(sharedBackend.io.mem_req(slot))
        }
      }
    }

    // Shared config arbiter: enabled accelerators -> 1 SharedMemBackend config port
    val cfgArb = Module(new Arbiter(new MemConfigerIO(cfg0), enabledAccelerators.size))
    for ((acc, i) <- enabledAccelerators.zipWithIndex) {
      cfgArb.io.in(i) <> acc.io.shared_config
    }
    sharedBackend.io.config <> cfgArb.io.out

    sharedBackend.io.query_vbank_id := enabledAccelerators.head.io.shared_query_vbank_id
    for (i <- 0 until nCores) {
      accelerators(i).foreach { acc =>
        acc.io.shared_query_group_count := sharedBackend.io.query_group_count
      }
    }

    // BarrierUnit (tile-level singleton)
    val barrierUnit = Module(new BarrierUnit(enabledAccelerators.size))
    for ((acc, i) <- enabledAccelerators.zipWithIndex) {
      barrierUnit.io.arrive(i) := acc.io.barrier_arrive
      acc.io.barrier_release   := barrierUnit.io.release(i)
    }

  } else {
    // No accelerator — tie off RoCC
    cores.foreach { c =>
      c.io.rocc.cmd.ready  := false.B
      c.io.rocc.resp.valid := false.B
      c.io.rocc.resp.bits  := DontCare
      c.io.rocc.busy       := DontCare
      c.io.rocc.interrupt  := DontCare
      c.io.rocc.mem        := DontCare
    }
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
  def injectBuildRoCC(p: Parameters, withBuckyball: Boolean, nCores: Int): Parameters =
    if (withBuckyball)
      p.alterPartial { case BuildRoCC => Seq.fill(nCores)((_: Parameters) => null.asInstanceOf[LazyRoCC]) }
    else p

}
