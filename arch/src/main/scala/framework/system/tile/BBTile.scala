package framework.system.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy.lazymodule._

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.diplomacy.{AddressSet, BufferParams, DisableMonitors, IdRange}
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
import freechips.rocketchip.tilelink.{TLBuffer, TLCacheCork, TLIdentityNode, TLWidthWidget, TLXbar}
import freechips.rocketchip.amba.axi4.{
  AXI4MasterNode,
  AXI4MasterParameters,
  AXI4MasterPortParameters,
  AXI4ToTL,
  AXI4UserYanker
}
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.prci.{ClockCrossingType, RationalCrossing}
import freechips.rocketchip.util.{Annotated, BooleanToAugmentedBoolean}

import sifive.blocks.inclusivecache.{CacheParameters, InclusiveCache, InclusiveCacheMicroParameters}

import framework.system.core.BBCore

/**
 * BBTile - Tile shell with Diplomacy boundary
 *
 * Architecture:
 * - Diplomacy layer: Frontend/HellaCache LazyModule per SM, optional L2
 * - Module layer: BBCore (pure Module)
 * - AXI4→TL conversion: RocketChip's AXI4ToTL LazyModule
 *
 * Phase 5 constraints (initial implementation):
 * - Single Core only (cores.size == 1)
 * - Single SM in that Core (cores(0).nSMs == 1)
 * - No L2 Cache
 * - No BEU (Bus Error Unit)
 * - No DTIM
 * - No SCU MMIO device
 * - No Vector unit
 *
 * Future Phase 6+ extensions:
 * - Multi-Core / multi-SM support (extra Frontend/HellaCache instances per SM)
 * - L2 cache integration (InclusiveCache)
 * - BEU/DTIM/SCU MMIO device integrations
 * - TileLink parameter alignment between BBCore hardcoded values and Diplomacy negotiation
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
    this(params, crossing.crossingType, lookup, BBTile.injectBuildRoCC(p, params.hasAnyBuckyball, params.totalSMs))

  // === Phase 6 constraints (relaxed from Phase 5) ===
  // Multi-Core and multi-SM now supported
  // L2 cache now supported (optional)
  require(bbParams.cores.forall(_.beuAddr.isEmpty), "Phase 6: BEU not yet supported")
  require(
    bbParams.cores.forall(c => c.scu.interceptMMIO == false || c.scu.interceptMMIO == true),
    "SCU intercept handled inside BBCore (passthrough for now)"
  )

  val coreParams   = bbParams.cores(0)
  val nCores       = bbParams.cores.size
  val totalSMs     = bbParams.totalSMs
  val hasBuckyball = bbParams.hasAnyBuckyball

  /**
   * Convert a global SM index (across all Cores) into (coreIdx, localSmIdx).
   *
   * Example: cores = [Core(2 SMs), Core(3 SMs)] -> totalSMs = 5
   *   globalSMId 0 -> (0, 0)
   *   globalSMId 1 -> (0, 1)
   *   globalSMId 2 -> (1, 0)
   *   globalSMId 3 -> (1, 1)
   *   globalSMId 4 -> (1, 2)
   */
  def globalSMToCoreSM(globalSMId: Int): (Int, Int) = {
    require(globalSMId >= 0 && globalSMId < totalSMs, s"globalSMId $globalSMId out of range [0, $totalSMs)")
    var remaining = globalSMId
    var coreIdx   = 0
    while (remaining >= bbParams.cores(coreIdx).nSMs) {
      remaining -= bbParams.cores(coreIdx).nSMs
      coreIdx += 1
    }
    (coreIdx, remaining)
  }

  // RoCC CSRs - Buckyball does not use custom CSRs
  val roccCSRs: Seq[Seq[CustomCSR]] = Nil

  // ---------------------------------------------------------------------------
  // Diplomacy nodes - tile boundary
  // ---------------------------------------------------------------------------
  val intOutwardNode = coreParams.beuAddr.map(_ => IntIdentityNode())
  val slaveNode      = TLIdentityNode()
  val masterNode     = visibilityNode

  // ---------------------------------------------------------------------------
  // Per-SM Frontend/HellaCache LazyModules
  //
  // HasHellaCache provides outer.dcache and HasICacheFrontend provides
  // outer.frontend - these are reserved for SM 0 (globally). For totalSMs > 1,
  // we create extra Frontend instances here (one per additional SM).
  //
  // The "primary" L1 (outer.frontend / outer.dcache) is automatically wired
  // into tlMasterXbar by the HasHellaCache / HasICacheFrontend mixins.
  //
  // DCache: For Phase 6, all SMs share the single outer.dcache via dcacheArb.
  // Multi-DCache support is deferred to a future phase.
  // ---------------------------------------------------------------------------
  val extraFrontends: Seq[Frontend] = (1 until totalSMs).map { smGlobalId =>
    val (coreIdx, _) = globalSMToCoreSM(smGlobalId)
    val icacheParams = bbParams.cores(coreIdx).l1ICache
    val f            = LazyModule(new Frontend(icacheParams, bbParams.tileId))
    tlMasterXbar.node                               := TLWidthWidget(icacheParams.rowBits / 8) := f.masterNode
    connectTLSlave(f.slaveNode, bbParams.cores(coreIdx).smParams.rocket.fetchBytes)
    f.icache.hartIdSinkNodeOpt.foreach(_            := hartIdNexusNode)
    f.icache.mmioAddressPrefixSinkNodeOpt.foreach(_ := mmioAddressPrefixNexusNode)
    f.resetVectorSinkNode                           := resetVectorNexusNode
    f
  }

  nPTWPorts += extraFrontends.size

  // ---------------------------------------------------------------------------
  // Buckyball DMA AXI4 master nodes + AXI4ToTL conversion
  //
  // BBCore outputs raw AXI4 bundles (io.axi_readers, io.axi_writers) for each
  // SM's Buckyball accelerator. We create AXI4MasterNode for each, then use
  // RocketChip's AXI4ToTL to convert to TileLink, aggregate with TLXbar,
  // and feed into tlOtherMastersNode.
  //
  // Topology per SM:
  //   BBCore.io.axi_readers[i] -> axi4ReaderNodes[i] -> AXI4ToTL -> TLXbar
  //   BBCore.io.axi_writers[i] -> axi4WriterNodes[i] -> AXI4ToTL -> TLXbar
  //                                                                    │
  //                                                                    ▼
  //                                                          tlOtherMastersNode
  // ---------------------------------------------------------------------------
  val (axi4ReaderNodes, axi4WriterNodes, bbDmaXbar) =
    if (hasBuckyball) {
      val dmaIdBits    = 2 // AXI4 ID bits for DMA (small to avoid source space explosion)
      val dmaMaxFlight = 4 // Max outstanding transactions per ID

      // Create AXI4MasterNode for each SM's reader and writer
      // aligned=true means transactions are aligned to size (required by AXI4ToTL)
      val readers = (0 until totalSMs).map { smId =>
        AXI4MasterNode(Seq(AXI4MasterPortParameters(
          masters = Seq(AXI4MasterParameters(
            name = s"bb-dma-reader-sm$smId",
            id = IdRange(0, 1 << dmaIdBits),
            aligned = true,
            maxFlight = Some(dmaMaxFlight)
          ))
        )))
      }

      val writers = (0 until totalSMs).map { smId =>
        AXI4MasterNode(Seq(AXI4MasterPortParameters(
          masters = Seq(AXI4MasterParameters(
            name = s"bb-dma-writer-sm$smId",
            id = IdRange(0, 1 << dmaIdBits),
            aligned = true,
            maxFlight = Some(dmaMaxFlight)
          ))
        )))
      }

      // Create AXI4ToTL converters and TLXbar to aggregate
      val xbar = TLXbar()
      readers.foreach { node =>
        xbar := AXI4ToTL() := AXI4UserYanker() := node
      }
      writers.foreach { node =>
        xbar := AXI4ToTL() := AXI4UserYanker() := node
      }

      (Some(readers), Some(writers), Some(xbar))
    } else {
      (None, None, None)
    }

  // Connect DMA xbar to tlOtherMastersNode if Buckyball is present
  bbDmaXbar.foreach { xbar =>
    tlOtherMastersNode := TLBuffer() := xbar
  }

  // ---------------------------------------------------------------------------
  // SCU: tile-local MMIO device, connected to tlSlaveXbar (like DTIM/BEU)
  // This allows CPU to access SCU via MMIO while Buckyball DMA goes directly
  // to memory via tlOtherMastersNode.
  // ---------------------------------------------------------------------------
  p(sims.scu.SCUKey).foreach { params =>
    val scu = LazyModule(new sims.scu.TLSCU(params, xBytes, bbParams.tileId))
    connectTLSlave(scu.node, xBytes)
  }

  // ---------------------------------------------------------------------------
  // Per-tile private L2 cache (optional)
  //
  // When bbParams.l2cache is defined, instantiate an InclusiveCache with
  // accompanying buffers (inner/outer) and a TLCacheCork. Topology becomes:
  //   tlOtherMastersNode -> innerBuf -> L2 -> outerBuf -> cork -> masterNode
  //
  // When bbParams.l2cache is None, the original direct connection is kept:
  //   tlOtherMastersNode -> masterNode
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
  //
  // Standard pattern: tlMasterXbar (L1 caches) -> tlOtherMastersNode -> masterNode
  // Buckyball DMA (AXI4→TL via AXI4ToTL) is also fed into tlOtherMastersNode.
  // When L2 is present, route through L2 between tlOtherMastersNode and masterNode.
  //
  // Use DisableMonitors on master paths to avoid TLMonitor mask validation
  // errors when Diplomacy inserts TLWidthWidget for AXI4 128-bit -> TL 64-bit
  // conversion (the widget has a known mask calculation bug).
  // ---------------------------------------------------------------------------
  DisableMonitors { implicit p =>
    tlOtherMastersNode := tlMasterXbar.node

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
        // Direct connection (no L2)
        masterNode :=* tlOtherMastersNode
    }
  }

  DisableMonitors(implicit p => tlSlaveXbar.node :*= slaveNode)

  // DCache port count: totalSMs (one dmem per SM) + Buckyball tieoff (if any)
  // Phase 6: All SMs share the single outer.dcache via dcacheArb
  nDCachePorts += totalSMs + hasBuckyball.toInt

  // Buckyball PTW port (one per Buckyball SM)
  if (hasBuckyball) {
    nPTWPorts += totalSMs
  }

  // ---------------------------------------------------------------------------
  // Device tree properties
  // ---------------------------------------------------------------------------
  val cpuDevice: SimpleDevice = new SimpleDevice("cpu", Seq("sifive,rocket0", "riscv")) {
    override def parent = Some(ResourceAnchors.cpus)

    override def describe(resources: ResourceBindings): Description = {
      val Description(name, mapping) = super.describe(resources)
      Description(name, mapping ++ cpuProperties ++ nextLevelCacheProperty ++ tileProperties)
    }

  }

  ResourceBinding {
    Resource(cpuDevice, "reg").bind(ResourceAddress(bbParams.tileId))
  }

  override lazy val module = new BBTileModuleImp(this)

  override def makeMasterBoundaryBuffers(crossing: ClockCrossingType)(implicit p: Parameters) =
    DisableMonitors { implicit p =>
      (bbParams.boundaryBuffers, crossing) match {
        case (Some(RocketTileBoundaryBufferParams(true)), _) => TLBuffer()
        case (Some(RocketTileBoundaryBufferParams(false)), _: RationalCrossing) =>
          TLBuffer(BufferParams.none, BufferParams.flow, BufferParams.none, BufferParams.flow, BufferParams(1))
        case _                                               => TLBuffer(BufferParams.none)
      }
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

  val nCores       = outer.nCores
  val totalSMs     = outer.totalSMs
  val hasBuckyball = outer.hasBuckyball

  // ---------------------------------------------------------------------------
  // Instantiate N BBCores (one per Core)
  // ---------------------------------------------------------------------------
  val bbcores = outer.bbParams.cores.zipWithIndex.map { case (coreParams, coreId) =>
    Module(new BBCore(coreParams, coreId)(outer.p))
  }

  // ---------------------------------------------------------------------------
  // Wire BBCore's AXI4 outputs to Diplomacy AXI4MasterNodes
  //
  // BBCore exposes raw AXI4 bundles for each SM's Buckyball DMA (reader + writer).
  // We connect them to the AXI4MasterNodes we created in the Diplomacy layer.
  // The AXI4ToTL converters and TLXbar handle the conversion to TileLink.
  // ---------------------------------------------------------------------------
  if (outer.hasBuckyball) {
    // Build per-Core flat SM index ranges
    var smGlobalOffset = 0
    bbcores.zipWithIndex.foreach { case (bbcore, coreId) =>
      val coreParams = outer.bbParams.cores(coreId)
      if (coreParams.smParams.hasBuckyball) {
        for (smIdx <- 0 until coreParams.nSMs) {
          val smGlobalId     = smGlobalOffset + smIdx
          val (axi_r_out, _) = outer.axi4ReaderNodes.get(smGlobalId).out(0)
          val (axi_w_out, _) = outer.axi4WriterNodes.get(smGlobalId).out(0)
          axi_r_out <> bbcore.io.axi_readers.get(smIdx)
          axi_w_out <> bbcore.io.axi_writers.get(smIdx)
        }
      }
      smGlobalOffset += coreParams.nSMs
    }
  }

  // ---------------------------------------------------------------------------
  // Wire BBCore <-> Frontend / HellaCache (Diplomacy LazyModule cache CPU ports)
  //
  // Each SM gets its own ICache (Frontend). SM 0 globally uses outer.frontend,
  // all others use outer.extraFrontends.
  //
  // DCache: Phase 6 shares the single outer.dcache among all SMs via dcacheArb.
  // ---------------------------------------------------------------------------

  // Wire each BBCore's IOs
  bbcores.zipWithIndex.foreach { case (bbcore, coreId) =>
    val coreParams = outer.bbParams.cores(coreId)

    // For each SM in this Core
    for (smIdx <- 0 until coreParams.nSMs) {
      val smGlobalId = outer.bbParams.cores.take(coreId).map(_.nSMs).sum + smIdx

      // ICache: connect to either outer.frontend (SM 0 globally) or extraFrontends
      val frontendModule =
        if (smGlobalId == 0) {
          outer.frontend.module
        } else {
          outer.extraFrontends(smGlobalId - 1).module
        }
      frontendModule.io.cpu <> bbcore.io.imem(smIdx)

      // DCache: add to dcachePorts (will be wired via dcacheArb later)
      dcachePorts += bbcore.io.dmem(smIdx)

      // PTW: SM 0 globally uses ptw.io.dpath, others tied off
      // (DatapathPTWIO is RocketCore-specific and only one PTW.io.dpath exists.
      //  Multi-SM support would require multi-PTW or PTW arbitration - TBD.)
      if (smGlobalId == 0) {
        bbcore.io.ptw_dpath(smIdx) <> ptw.io.dpath
      } else {
        bbcore.io.ptw_dpath(smIdx) := DontCare
      }

      // Buckyball PTW (if this Core has Buckyball)
      if (coreParams.smParams.hasBuckyball) {
        val bbPtw  = Wire(new TLBPTWIO)
        ptwPorts += bbPtw
        val accPtw = bbcore.io.ptw_buckyball.get(smIdx)

        // Wire BBTLBPTWIO <-> TLBPTWIO field-by-field
        bbPtw.req.valid              := accPtw.req.valid
        bbPtw.req.bits.valid         := accPtw.req.bits.valid
        bbPtw.req.bits.bits.addr     := accPtw.req.bits.bits.addr
        bbPtw.req.bits.bits.need_gpa := accPtw.req.bits.bits.need_gpa
        bbPtw.req.bits.bits.vstage1  := accPtw.req.bits.bits.vstage1
        bbPtw.req.bits.bits.stage2   := accPtw.req.bits.bits.stage2
        accPtw.req.ready             := bbPtw.req.ready

        accPtw.resp.valid                          := bbPtw.resp.valid
        accPtw.resp.bits.ae_ptw                    := bbPtw.resp.bits.ae_ptw
        accPtw.resp.bits.ae_final                  := bbPtw.resp.bits.ae_final
        accPtw.resp.bits.pf                        := bbPtw.resp.bits.pf
        accPtw.resp.bits.gf                        := bbPtw.resp.bits.gf
        accPtw.resp.bits.hr                        := bbPtw.resp.bits.hr
        accPtw.resp.bits.hw                        := bbPtw.resp.bits.hw
        accPtw.resp.bits.hx                        := bbPtw.resp.bits.hx
        accPtw.resp.bits.pte.ppn                   := bbPtw.resp.bits.pte.ppn
        accPtw.resp.bits.pte.reserved_for_future   := bbPtw.resp.bits.pte.reserved_for_future
        accPtw.resp.bits.pte.reserved_for_software := bbPtw.resp.bits.pte.reserved_for_software
        accPtw.resp.bits.pte.d                     := bbPtw.resp.bits.pte.d
        accPtw.resp.bits.pte.a                     := bbPtw.resp.bits.pte.a
        accPtw.resp.bits.pte.g                     := bbPtw.resp.bits.pte.g
        accPtw.resp.bits.pte.u                     := bbPtw.resp.bits.pte.u
        accPtw.resp.bits.pte.x                     := bbPtw.resp.bits.pte.x
        accPtw.resp.bits.pte.w                     := bbPtw.resp.bits.pte.w
        accPtw.resp.bits.pte.r                     := bbPtw.resp.bits.pte.r
        accPtw.resp.bits.pte.v                     := bbPtw.resp.bits.pte.v
        accPtw.resp.bits.level                     := bbPtw.resp.bits.level
        accPtw.resp.bits.fragmented_superpage      := bbPtw.resp.bits.fragmented_superpage
        accPtw.resp.bits.homogeneous               := bbPtw.resp.bits.homogeneous
        accPtw.resp.bits.gpa.valid                 := bbPtw.resp.bits.gpa.valid
        accPtw.resp.bits.gpa.bits                  := bbPtw.resp.bits.gpa.bits
        accPtw.resp.bits.gpa_is_pte                := bbPtw.resp.bits.gpa_is_pte

        accPtw.ptbr.mode  := bbPtw.ptbr.mode
        accPtw.ptbr.asid  := bbPtw.ptbr.asid
        accPtw.ptbr.ppn   := bbPtw.ptbr.ppn
        accPtw.hgatp.mode := bbPtw.hgatp.mode
        accPtw.hgatp.asid := bbPtw.hgatp.asid
        accPtw.hgatp.ppn  := bbPtw.hgatp.ppn
        accPtw.vsatp.mode := bbPtw.vsatp.mode
        accPtw.vsatp.asid := bbPtw.vsatp.asid
        accPtw.vsatp.ppn  := bbPtw.vsatp.ppn
        accPtw.status     := bbPtw.status
        accPtw.hstatus    := bbPtw.hstatus
        accPtw.gstatus    := bbPtw.gstatus
        accPtw.pmp.zipWithIndex.foreach { case (pmpPort, j) =>
          pmpPort.cfg.l   := bbPtw.pmp(j).cfg.l
          pmpPort.cfg.res := bbPtw.pmp(j).cfg.res
          pmpPort.cfg.a   := bbPtw.pmp(j).cfg.a
          pmpPort.cfg.x   := bbPtw.pmp(j).cfg.x
          pmpPort.cfg.w   := bbPtw.pmp(j).cfg.w
          pmpPort.cfg.r   := bbPtw.pmp(j).cfg.r
          pmpPort.addr    := bbPtw.pmp(j).addr
          pmpPort.mask    := bbPtw.pmp(j).mask
        }
        accPtw.customCSRs := DontCare
        bbPtw.customCSRs  := DontCare
      }

      // Hartid
      bbcore.io.hartids(smIdx)       := outer.hartIdSinkNode.bundle + smGlobalId.U
      bbcore.io.reset_vectors(smIdx) := DontCare // overridden by reset_vector_nexus

      // Interrupts
      outer.decodeCoreInterrupts(bbcore.io.interrupts(smIdx))
      bbcore.io.interrupts(smIdx).nmi.foreach(nmi => nmi := outer.nmiSinkNode.get.bundle)
    }

    // Per-Core signals
    bbcore.io.traceStall := outer.traceAuxSinkNode.bundle.stall
  }

  // ---------------------------------------------------------------------------
  // Trace / cease / WFI (aggregate across all Cores)
  // ---------------------------------------------------------------------------
  outer.traceSourceNode.bundle <> bbcores(0).io.traces(0)
  outer.bpwatchSourceNode.bundle := DontCare // BBCore does not currently expose bpwatch

  val allCease                = bbcores.flatMap(_.io.cease)
  val allFrontendClockEnabled = (outer.frontend.module +: outer.extraFrontends.map(_.module)).map(_.io.cpu.clock_enabled)

  outer.reportCease(outer.bbParams.cores(0).smParams.enableClockGate.option(
    !outer.dcache.module.io.cpu.clock_enabled &&
      !allFrontendClockEnabled.reduce(_ || _) &&
      !ptw.io.dpath.clock_enabled &&
      allCease.reduce(_ && _)
  ))
  outer.reportWFI(Some(bbcores.flatMap(_.io.wfi).reduce(_ && _)))
  outer.reportHalt(List(outer.dcache.module.io.errors))

  // ---------------------------------------------------------------------------
  // DCache arbiter: all SMs' dmem ports have been added to dcachePorts above
  // ---------------------------------------------------------------------------

  // RoCC mem tieoff (one slot when Buckyball is present, matching nDCachePorts)
  if (hasBuckyball) {
    val roccMemIF = Module(new SimpleHellaCacheIF())
    roccMemIF.io.requestor.req.valid          := false.B
    roccMemIF.io.requestor.req.bits           := DontCare
    roccMemIF.io.requestor.s1_kill            := false.B
    roccMemIF.io.requestor.s1_data            := DontCare
    roccMemIF.io.requestor.s2_kill            := false.B
    roccMemIF.io.requestor.keep_clock_enabled := false.B
    dcachePorts += roccMemIF.io.cache
  }

  // ---------------------------------------------------------------------------
  // Finalize DCache arbiter and PTW connections
  // ---------------------------------------------------------------------------
  val h = dcachePorts.size
  val o = outer.nDCachePorts
  require(h == o, s"dcachePorts size was $h, outer.nDCachePorts expected $o")

  dcacheArb.io.requestor <> dcachePorts.toSeq
  ptw.io.requestor <> ptwPorts.toSeq
}

// =============================================================================
// Companion object - utility helpers
// =============================================================================
object BBTile {

  /**
   * Inject a dummy BuildRoCC entry so that usingRoCC=true throughout all
   * HasRocketCoreParameters mixins (CSR, decode, etc.), without actually
   * using the LazyRoCC mechanism.
   *
   * BBCore manages RoCC connections internally between RocketCore and
   * BuckyballAccelerator (no LazyRoCC needed).
   */
  def injectBuildRoCC(p: Parameters, withBuckyball: Boolean, totalSMs: Int): Parameters =
    if (withBuckyball)
      p.alterPartial { case BuildRoCC =>
        Seq.fill(totalSMs)((_: Parameters) => null.asInstanceOf[LazyRoCC])
      }
    else p

}
