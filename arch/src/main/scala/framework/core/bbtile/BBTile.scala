package framework.core.bbtile

import chisel3._
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

/**
 * BBTile — a composable tile containing Rocket core(s) + optional Buckyball accelerator(s).
 *
 * Design principles:
 *   - Extends BaseTile for CanAttachTile compatibility (diplomacy shell)
 *   - Mixes in HasHellaCache + HasICacheFrontend for Rocket core infrastructure (unavoidable)
 *   - Buckyball accelerator is composed as an independent module, NOT via LazyRoCCBB inheritance
 *   - RoCC cmd/resp are wired directly inside the tile
 *   - Accelerator TileLink DMA nodes are declared in the diplomacy shell
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
  // Buckyball accelerator TileLink nodes (diplomacy layer)
  // ---------------------------------------------------------------------------
  val bbConfig = bbParams.buckyballConfig

  val bb_reader_node =
    if (bbParams.withBuckyball) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = "bb-dma-reader",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbConfig.memDomain.dma_n_xacts)
    ))))))
    else None

  val bb_writer_node =
    if (bbParams.withBuckyball) Some(TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
      name = "bb-dma-writer",
      sourceId = freechips.rocketchip.diplomacy.IdRange(0, bbConfig.memDomain.dma_n_xacts)
    ))))))
    else None

  val bb_xbar_node =
    if (bbParams.withBuckyball) {
      val xbar = TLXbar()
      xbar := TLBuffer() := bb_reader_node.get
      xbar := TLBuffer() := bb_writer_node.get
      Some(xbar)
    } else None

  // Connect Buckyball DMA to tile master port (through width widget)
  bb_xbar_node.foreach { xbar =>
    tlOtherMastersNode :=* TLWidthWidget(bbConfig.memDomain.dma_buswidth / 8) := TLBuffer() := xbar
  }

  // ---------------------------------------------------------------------------
  // TileLink topology
  // ---------------------------------------------------------------------------
  tlOtherMastersNode := tile_master_blocker.map(_.node := tlMasterXbar.node).getOrElse(tlMasterXbar.node)
  masterNode :=* tlOtherMastersNode
  DisableMonitors(implicit p => tlSlaveXbar.node :*= slaveNode)

  // DCache port count: core + PTW(via usingVM) + DTIM + vector
  nDCachePorts += 1 + (dtim_adapter.isDefined).toInt +
    bbParams.core.vector.map(_.useDCache.toInt).getOrElse(0) +
    bbParams.withBuckyball.toInt // RoCC mem port (tied off but counted by dcacheArbPorts)

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

  // Buckyball needs one PTW port for its TLB
  if (bbParams.withBuckyball) {
    nPTWPorts += 1
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
  // Buckyball accelerator (composed, not inherited via LazyRoCCBB)
  // ---------------------------------------------------------------------------
  if (outer.bbParams.withBuckyball) {
    val (tl_reader, edge_reader) = outer.bb_reader_node.get.out(0)
    val (tl_writer, _)           = outer.bb_writer_node.get.out(0)

    val buckyball = Module(new BuckyballAccelerator(outer.bbConfig)(edge_reader))

    // RoCC cmd/resp: direct wiring (both use RoCCCommandBB/RoCCResponseBB)
    buckyball.io.cmd <> core.io.rocc.cmd
    core.io.rocc.resp <> buckyball.io.resp
    core.io.rocc.busy      := buckyball.io.busy
    core.io.rocc.interrupt := buckyball.io.interrupt

    // DMA TileLink
    tl_reader <> buckyball.io.tl_reader
    tl_writer <> buckyball.io.tl_writer

    // PTW: Buckyball's BBTLBPTWIO <-> tile's TLBPTWIO (field-by-field adaptation)
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

    // TLB exception
    buckyball.io.tlbExp(0).flush_skip  := false.B
    buckyball.io.tlbExp(0).flush_retry := false.B

    // CPU sfence → Buckyball TLB flush
    buckyball.io.sfence := ptw.io.dpath.sfence.valid

    // RoCC mem: Buckyball doesn't use the HellaCacheIO mem port, but the
    // DCache arbiter still expects a port for it (dcacheArbPorts counts BuildRoCC.size).
    // Route through SimpleHellaCacheIF with tied-off requestor side.
    val roccMemIF = Module(new SimpleHellaCacheIF())
    roccMemIF.io.requestor.req.valid          := false.B
    roccMemIF.io.requestor.req.bits           := DontCare
    roccMemIF.io.requestor.s1_kill            := false.B
    roccMemIF.io.requestor.s1_data            := DontCare
    roccMemIF.io.requestor.s2_kill            := false.B
    roccMemIF.io.requestor.keep_clock_enabled := false.B
    dcachePorts += roccMemIF.io.cache
    core.io.rocc.mem                          := DontCare
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
