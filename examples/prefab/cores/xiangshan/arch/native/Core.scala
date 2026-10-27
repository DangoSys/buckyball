// Copied from XiangShan src/main/scala/xiangshan/XSCore.scala and modified for Buckyball.
// XiangShan is licensed under Mulan PSL v2 (http://license.coscl.org.cn/MulanPSL2).

package examples.core.xiangshan.native

import org.chipsalliance.cde.config
import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import coupledL2.PrefetchCtrlFromCore
import freechips.rocketchip.diplomacy.{BundleBridgeSource, LazyModule, LazyModuleImp}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile.HasFPUParameters
import system.{HasSoCParameter, SoCParamsKey}
import utils._
import utility._
import utility.mbist.{MbistInterface, MbistPipeline}
import utility.sram.{SramBroadcastBundle, SramHelper}
import xiangshan.frontend._
import xiangshan.backend._
import xiangshan.backend.fu.PMPRespBundle
import xiangshan.backend.trace.TraceCoreInterface
import xiangshan.mem._
import xiangshan.cache.mmu._
import xiangshan.cache.mmu.TlbRequestIO
import scala.collection.mutable.ListBuffer
import xiangshan._
import examples.core.xiangshan.native.{Backend, Commands}

abstract class XSCoreBase()(implicit p: config.Parameters) extends LazyModule with HasXSParameter {
  override def shouldBeInlined: Boolean = false
  // outer facing nodes
  val frontend = LazyModule(new Frontend())
  val csrOut   = BundleBridgeSource(Some(() => new DistributedCSRIO()))
  val backend  = LazyModule(new Backend(backendParams))

  val memBlock = LazyModule(new MemBlock)

  memBlock.inner.frontendBridge.icache_node        := frontend.inner.icache.clientNode
  memBlock.inner.frontendBridge.instr_uncache_node := frontend.inner.instrUncache.clientNode
  if (p(SoCParamsKey).EnableICacheCtrl) {
    frontend.inner.icache.ctrlUnitOpt.get.node := memBlock.inner.frontendBridge.icachectrl_node
  }
}

class XSCore()(implicit p: config.Parameters) extends XSCoreBase with HasXSDts {
  lazy val module = new XSCoreImp(this)
}

class XSCoreImp(outer: XSCoreBase) extends LazyModuleImp(outer) with HasXSParameter with HasSoCParameter {

  val io = IO(new Bundle {
    val hartId             = Input(UInt(hartIdLen.W))
    val msiInfo            = Input(ValidIO(UInt(soc.IMSICParams.MSI_INFO_WIDTH.W)))
    val msiAck             = Output(Bool())
    val teemsiInfo         = Option.when(soc.IMSICParams.HasTEEIMSIC)(Input(ValidIO(UInt(soc.IMSICParams.MSI_INFO_WIDTH.W))))
    val teemsiAck          = Option.when(soc.IMSICParams.HasTEEIMSIC)(Output(Bool()))
    val commands           = new Commands
    val timeRequest        = Decoupled(Bool())
    val clintTime          = Input(ValidIO(UInt(64.W)))
    val reset_vector       = Input(UInt(PAddrBits.W))
    val reset_mtvec        = Option.when(enableResetMtvec)(Input(UInt(PAddrBits.W)))
    val cpu_halt           = Output(Bool())
    val l2_flush_done      = Input(Bool())
    val l2_flush_en        = Output(Bool())
    val power_down_en      = Output(Bool())
    val cpu_critical_error = Output(Bool())
    val resetInFrontend    = Output(Bool())
    val traceCoreInterface = new TraceCoreInterface
    val l2PfCtrl           = Output(new PrefetchCtrlFromCore)
    val perfEvents         = Input(Vec(numPCntHc * coreParams.L2NBanks + 1, new PerfEvent))
    val beu_errors         = Output(new XSL1BusErrors())
    val l2_hint            = Input(Valid(new L2ToL1Hint()))
    val l2_tlb_req         = Flipped(new TlbRequestIO(nRespDups = 2))
    val l2_pmp_resp        = new PMPRespBundle
    val l2PfqBusy          = Input(Bool())

    val debugTopDown = new Bundle {
      val robTrueCommit = Output(UInt(64.W))
      val robHeadPaddr  = Valid(UInt(PAddrBits.W))
      val l2MissMatch   = Input(Bool())
      val l3MissMatch   = Input(Bool())
    }

    val topDownInfo = Input(new Bundle {
      val l2Miss = Bool()
      val l3Miss = Bool()
    })

    val dft       = Option.when(hasDFT)(Input(new SramBroadcastBundle))
    val dft_reset = Option.when(hasDFT)(Input(new DFTResetSignals()))
  })

  dontTouch(io.l2_flush_done)
  dontTouch(io.l2_flush_en)
  dontTouch(io.power_down_en)

  println(s"FPGAPlatform:${env.FPGAPlatform} EnableDebug:${env.EnableDebug}")

  val frontend = outer.frontend.module
  val backend  = outer.backend.module
  io.commands <> backend.io.commands
  io.timeRequest <> backend.io.timeRequest
  val memBlock = outer.memBlock.module

  frontend.io.hartId       := memBlock.io.inner_hartId
  frontend.io.reset_vector := memBlock.io.inner_reset_vector
  frontend.io.softPrefetch <> memBlock.io.ifetchPrefetch
  frontend.io.backend <> backend.io.frontend
  frontend.io.sfence <> backend.io.frontendSfence
  frontend.io.tlbCsr <> backend.io.frontendTlbCsr
  frontend.io.csrCtrl <> backend.io.frontendCsrCtrl
  frontend.io.fencei <> backend.io.fenceio.fencei

  backend.io.fromTop := memBlock.io.mem_to_ooo.topToBackendBypass

  require(backend.io.mem.stIn.length == memBlock.io.mem_to_ooo.stIn.length)
  backend.io.mem.stIn.zip(memBlock.io.mem_to_ooo.stIn).foreach { case (sink, source) =>
    sink.valid            := source.valid
    sink.bits             := 0.U.asTypeOf(sink.bits)
    sink.bits.robIdx      := source.bits.uop.robIdx
    sink.bits.ssid        := source.bits.uop.ssid
    sink.bits.storeSetHit := source.bits.uop.storeSetHit
  // The other signals have not been used
  }
  backend.io.mem.memoryViolation       := memBlock.io.mem_to_ooo.memoryViolation
  backend.io.mem.lsqEnqIO <> memBlock.io.ooo_to_mem.enqLsq
  backend.io.mem.sqDeq                 := memBlock.io.mem_to_ooo.sqDeq
  backend.io.mem.lqDeq                 := memBlock.io.mem_to_ooo.lqDeq
  backend.io.mem.sqDeqPtr              := memBlock.io.mem_to_ooo.sqDeqPtr
  backend.io.mem.lqDeqPtr              := memBlock.io.mem_to_ooo.lqDeqPtr
  backend.io.mem.lqCancelCnt           := memBlock.io.mem_to_ooo.lqCancelCnt
  backend.io.mem.sqCancelCnt           := memBlock.io.mem_to_ooo.sqCancelCnt
  backend.io.mem.otherFastWakeup       := memBlock.io.mem_to_ooo.otherFastWakeup
  backend.io.mem.stIssuePtr            := memBlock.io.mem_to_ooo.stIssuePtr
  backend.io.mem.ldaIqFeedback         := memBlock.io.mem_to_ooo.ldaIqFeedback
  backend.io.mem.staIqFeedback         := memBlock.io.mem_to_ooo.staIqFeedback
  backend.io.mem.hyuIqFeedback         := memBlock.io.mem_to_ooo.hyuIqFeedback
  backend.io.mem.vstuIqFeedback        := memBlock.io.mem_to_ooo.vstuIqFeedback
  backend.io.mem.vlduIqFeedback        := memBlock.io.mem_to_ooo.vlduIqFeedback
  backend.io.mem.ldCancel              := memBlock.io.mem_to_ooo.ldCancel
  backend.io.mem.wakeup                := memBlock.io.mem_to_ooo.wakeup
  backend.io.mem.writebackLda <> memBlock.io.mem_to_ooo.writebackLda
  backend.io.mem.writebackSta <> memBlock.io.mem_to_ooo.writebackSta
  backend.io.mem.writebackHyuLda <> memBlock.io.mem_to_ooo.writebackHyuLda
  backend.io.mem.writebackHyuSta <> memBlock.io.mem_to_ooo.writebackHyuSta
  backend.io.mem.writebackStd <> memBlock.io.mem_to_ooo.writebackStd
  backend.io.mem.writebackVldu <> memBlock.io.mem_to_ooo.writebackVldu
  backend.io.mem.robLsqIO.loadMmio     := memBlock.io.mem_to_ooo.lsqio.loadMmio
  backend.io.mem.robLsqIO.loadMmioUop  := memBlock.io.mem_to_ooo.lsqio.loadMmioUop
  backend.io.mem.robLsqIO.storeMmio    := memBlock.io.mem_to_ooo.lsqio.storeMmio
  backend.io.mem.robLsqIO.storeMmioUop := memBlock.io.mem_to_ooo.lsqio.storeMmioUop

  // memblock error exception writeback, 1 cycle after normal writeback
  backend.io.mem.s3_delayed_load_error := memBlock.io.mem_to_ooo.s3_delayed_load_error

  backend.io.mem.exceptionAddr.vaddr             := memBlock.io.mem_to_ooo.lsqio.vaddr
  backend.io.mem.exceptionAddr.gpaddr            := memBlock.io.mem_to_ooo.lsqio.gpaddr
  backend.io.mem.exceptionAddr.isForVSnonLeafPTE := memBlock.io.mem_to_ooo.lsqio.isForVSnonLeafPTE
  backend.io.mem.debugLS                         := memBlock.io.debug_ls
  backend.io.mem.lsTopdownInfo                   := memBlock.io.mem_to_ooo.lsTopdownInfo
  backend.io.mem.lqCanAccept                     := memBlock.io.mem_to_ooo.lsqio.lqCanAccept
  backend.io.mem.sqCanAccept                     := memBlock.io.mem_to_ooo.lsqio.sqCanAccept
  backend.io.fenceio.sbuffer.sbIsEmpty           := memBlock.io.mem_to_ooo.sbIsEmpty

  backend.io.perf.frontendInfo       := frontend.io.frontendInfo
  backend.io.perf.memInfo            := memBlock.io.memInfo
  backend.io.perf.perfEventsFrontend := frontend.io_perf
  backend.io.perf.perfEventsLsu      := memBlock.io_perf
  backend.io.perf.perfEventsHc       := memBlock.io.inner_hc_perfEvents
  backend.io.perf.perfEventsBackend  := DontCare
  backend.io.perf.retiredInstr       := DontCare
  backend.io.perf.ctrlInfo           := DontCare

  backend.io.mem.storeDebugInfo <> memBlock.io.mem_to_ooo.storeDebugInfo

  // top -> memBlock
  memBlock.io.fromTopToBackend.clintTime             := io.clintTime
  memBlock.io.fromTopToBackend.msiInfo               := io.msiInfo
  memBlock.io.fromTopToBackend.teemsiInfo zip io.teemsiInfo foreach { case (memBlock_teemsiInfo, io_teemsiInfo) =>
    memBlock_teemsiInfo := io_teemsiInfo
  }
  memBlock.io.hartId                                 := io.hartId
  memBlock.io.l2_flush_done                          := io.l2_flush_done
  memBlock.io.outer_reset_vector                     := io.reset_vector
  memBlock.io.fromTopToBackend.reset_mtvec.foreach(_ := io.reset_mtvec.get)
  memBlock.io.outer_hc_perfEvents                    := io.perfEvents
  // frontend -> memBlock
  memBlock.io.inner_beu_errors_icache <> frontend.io.error
  memBlock.io.ooo_to_mem.backendToTopBypass          := backend.io.toTop
  memBlock.io.ooo_to_mem.issueLda <> backend.io.mem.issueLda
  memBlock.io.ooo_to_mem.issueSta <> backend.io.mem.issueSta
  memBlock.io.ooo_to_mem.issueStd <> backend.io.mem.issueStd
  memBlock.io.ooo_to_mem.issueHya <> backend.io.mem.issueHylda
  backend.io.mem.issueHysta.foreach(_.ready          := false.B) // this fake port should not be used
  memBlock.io.ooo_to_mem.issueVldu <> backend.io.mem.issueVldu

  // By default, instructions do not have exceptions when they enter the function units.
  memBlock.io.ooo_to_mem.issueUops.map(_.bits.uop.clearExceptions())
  memBlock.io.ooo_to_mem.storePc          := backend.io.mem.storePcRead
  memBlock.io.ooo_to_mem.hybridPc         := backend.io.mem.hyuPcRead
  memBlock.io.ooo_to_mem.flushSb          := backend.io.fenceio.sbuffer.flushSb
  memBlock.io.ooo_to_mem.loadFastMatch    := 0.U.asTypeOf(memBlock.io.ooo_to_mem.loadFastMatch)
  memBlock.io.ooo_to_mem.loadFastImm      := 0.U.asTypeOf(memBlock.io.ooo_to_mem.loadFastImm)
  memBlock.io.ooo_to_mem.loadFastFuOpType := 0.U.asTypeOf(memBlock.io.ooo_to_mem.loadFastFuOpType)

  memBlock.io.ooo_to_mem.sfence <> backend.io.mem.sfence

  memBlock.io.redirect                        := backend.io.mem.redirect
  memBlock.io.ooo_to_mem.csrCtrl              := backend.io.mem.csrCtrl
  memBlock.io.ooo_to_mem.tlbCsr               := backend.io.mem.tlbCsr
  memBlock.io.ooo_to_mem.lsqio.lcommit        := backend.io.mem.robLsqIO.lcommit
  memBlock.io.ooo_to_mem.lsqio.scommit        := backend.io.mem.robLsqIO.scommit
  memBlock.io.ooo_to_mem.lsqio.pendingMMIOld  := backend.io.mem.robLsqIO.pendingMMIOld
  memBlock.io.ooo_to_mem.lsqio.pendingld      := backend.io.mem.robLsqIO.pendingld
  memBlock.io.ooo_to_mem.lsqio.pendingst      := backend.io.mem.robLsqIO.pendingst
  memBlock.io.ooo_to_mem.lsqio.pendingVst     := backend.io.mem.robLsqIO.pendingVst
  memBlock.io.ooo_to_mem.lsqio.commit         := backend.io.mem.robLsqIO.commit
  memBlock.io.ooo_to_mem.lsqio.pendingPtr     := backend.io.mem.robLsqIO.pendingPtr
  memBlock.io.ooo_to_mem.lsqio.pendingPtrNext := backend.io.mem.robLsqIO.pendingPtrNext
  memBlock.io.ooo_to_mem.isStoreException     := backend.io.mem.isStoreException
  memBlock.io.ooo_to_mem.isVlsException       := backend.io.mem.isVlsException

  memBlock.io.fetch_to_mem.itlb <> frontend.io.ptw
  memBlock.io.l2_hint.valid          := io.l2_hint.valid
  memBlock.io.l2_hint.bits.sourceId  := io.l2_hint.bits.sourceId
  memBlock.io.l2_tlb_req <> io.l2_tlb_req
  memBlock.io.l2_pmp_resp <> io.l2_pmp_resp
  memBlock.io.l2_hint.bits.isKeyword := io.l2_hint.bits.isKeyword
  memBlock.io.l2PfqBusy              := io.l2PfqBusy

  // if l2 prefetcher use stream prefetch, it should be placed in XSCore

  // top-down info
  memBlock.io.debugTopDown.robHeadVaddr        := backend.io.debugTopDown.fromRob.robHeadVaddr
  frontend.io.debugTopDown.robHeadVaddr        := backend.io.debugTopDown.fromRob.robHeadVaddr
  io.debugTopDown.robHeadPaddr                 := backend.io.debugTopDown.fromRob.robHeadPaddr
  io.debugTopDown.robTrueCommit                := backend.io.debugRolling.robTrueCommit
  backend.io.debugTopDown.fromCore.l2MissMatch := io.debugTopDown.l2MissMatch
  backend.io.debugTopDown.fromCore.l3MissMatch := io.debugTopDown.l3MissMatch
  backend.io.debugTopDown.fromCore.fromMem     := memBlock.io.debugTopDown.toCore
  memBlock.io.debugRolling                     := backend.io.debugRolling

  io.cpu_halt           := memBlock.io.outer_cpu_halt
  io.l2_flush_en        := memBlock.io.outer_l2_flush_en
  io.power_down_en      := memBlock.io.outer_power_down_en
  io.cpu_critical_error := memBlock.io.outer_cpu_critical_error
  io.msiAck             := memBlock.io.outer_msi_ack
  io.teemsiAck zip memBlock.io.outer_teemsi_ack foreach { case (io_teemsiAck, memBlock_outer_teemsi_ack) =>
    io_teemsiAck := memBlock_outer_teemsi_ack
  }
  io.beu_errors.icache <> memBlock.io.outer_beu_errors_icache
  io.beu_errors.dcache <> memBlock.io.dcacheError
  io.beu_errors.uncache <> memBlock.io.uncacheError
  io.beu_errors.l2 <> DontCare
  io.l2PfCtrl           := memBlock.io.outer_l2PfCtrl

  memBlock.io.resetInFrontendBypass.fromFrontend := frontend.io.resetInFrontend
  io.resetInFrontend                             := memBlock.io.resetInFrontendBypass.toL2Top
  memBlock.io.traceCoreInterfaceBypass.fromBackend <> backend.io.traceCoreInterface
  io.traceCoreInterface <> memBlock.io.traceCoreInterfaceBypass.toL2Top
  memBlock.io.wfi <> backend.io.mem.wfi
  memBlock.io.topDownInfo.fromL2Top.l2Miss       := io.topDownInfo.l2Miss
  memBlock.io.topDownInfo.fromL2Top.l3Miss       := io.topDownInfo.l3Miss
  backend.io.topDownInfo.replayAllocate          := memBlock.io.topDownInfo.toBackend.replayAllocate
  backend.io.topDownInfo.sqFull                  := memBlock.io.topDownInfo.toBackend.sqFull
  backend.io.topDownInfo.sbFull                  := memBlock.io.topDownInfo.toBackend.sbFull
  backend.io.topDownInfo.l1Miss                  := memBlock.io.topDownInfo.toBackend.l1Miss
  backend.io.topDownInfo.l2TopMiss.l2Miss        := memBlock.io.topDownInfo.toBackend.l2TopMiss.l2Miss
  backend.io.topDownInfo.l2TopMiss.l3Miss        := memBlock.io.topDownInfo.toBackend.l2TopMiss.l3Miss

  if (debugOpts.ResetGen) {
    backend.reset  := memBlock.io.reset_backend
    frontend.reset := backend.io.frontendReset
  }

  memBlock.io.dft.zip(io.dft).foreach({ case (a, b) => a := b })
  memBlock.io.dft_reset.zip(io.dft_reset).foreach({ case (a, b) => a := b })
  frontend.io.dft.zip(memBlock.io.dft_frnt).foreach({ case (a, b) => a := b })
  frontend.io.dft_reset.zip(memBlock.io.dft_reset_frnt).foreach({ case (a, b) => a := b })
  backend.io.dft.zip(memBlock.io.dft_bcknd).foreach({ case (a, b) => a := b })
  backend.io.dft_reset.zip(memBlock.io.dft_reset_bcknd).foreach({ case (a, b) => a := b })
}

trait HasXSDts {
  this: XSCore =>

  val device: SimpleDevice = new SimpleDevice("cpu", Seq("ICT,xiangshan", "riscv")) {
    override def parent: Some[Device] = Some(ResourceAnchors.cpus)

    def cpuProperties: PropertyMap = Map(
      "device_type"          -> "cpu".asProperty,
      "status"               -> "okay".asProperty,
      "clock-frequency"      -> 0.asProperty,
      "riscv,isa"            -> "rv64imafdcvh".asProperty, // deprecated
      "riscv,isa-base"       -> ISABase.asProperty,
      "riscv,isa-extensions" -> ISAExtensions.map(ResourceString),
      "timebase-frequency"   -> 1000000.asProperty
    )

    def tileProperties: PropertyMap = {
      val dcache =
        if (coreParams.dcacheParametersOpt.nonEmpty) Map(
          "d-cache-block-size" -> dcacheParameters.blockBytes.asProperty,
          "d-cache-sets"       -> dcacheParameters.nSets.asProperty,
          "d-cache-size"       -> (dcacheParameters.nSets * dcacheParameters.nWays * dcacheParameters.blockBytes).asProperty
        )
        else Map()

      val icache = Map(
        "i-cache-block-size" -> icacheParameters.blockBytes.asProperty,
        "i-cache-sets"       -> icacheParameters.nSets.asProperty,
        "i-cache-size"       -> (icacheParameters.nSets * icacheParameters.nWays * icacheParameters.blockBytes).asProperty
      )

      val dtlb = Map(
        "d-tlb-size" -> (ldtlbParams.NSets * ldtlbParams.NWays).asProperty,
        "d-tlb-sets" -> 1.asProperty
      )

      val itlb = Map(
        "i-tlb-size" -> (itlbParams.NSets * itlbParams.NWays).asProperty,
        "i-tlb-sets" -> 1.asProperty
      )

      val mmu = Map(
        "tlb-split" -> Nil,
        "mmu-type"  -> s"riscv,sv$VAddrBits".asProperty
      )

      val pmp = Nil

      dcache ++ icache ++ dtlb ++ itlb ++ mmu ++ pmp
    }

    def nextLevelCacheProperty: PropertyOption = {
      if (coreParams.dcacheParametersOpt.nonEmpty) {
        val outer = memBlock.inner.dcache.clientNode.edges.out.flatMap(_.manager.managers)
          .filter(_.supportsAcquireB)
          .flatMap(_.resources.headOption)
          .map(_.owner.label)
          .distinct
        if (outer.isEmpty) None
        else Some("next-level-cache" -> outer.map(l => ResourceReference(l)).toList)
      } else None
    }

    override def describe(resources: ResourceBindings): Description = {
      val Description(name, mapping) = super.describe(resources)
      Description(name, mapping ++ cpuProperties ++ nextLevelCacheProperty ++ tileProperties)
    }

  }

  val intcDevice = new DeviceSnippet {
    override def parent = Some(device)

    def describe(): Description = {
      Description(
        "interrupt-controller",
        Map(
          "compatible"           -> "riscv,cpu-intc".asProperty,
          "interrupt-controller" -> Nil,
          "#interrupt-cells"     -> 1.asProperty
        )
      )
    }

  }

  ResourceBinding {
    Resource(device, "reg").bind(ResourceAddress(coreParams.HartId))
    val int_resources = (
      memBlock.inner.clint_int_sink.edges.in.flatMap(_.source.sources) ++
        memBlock.inner.plic_int_sink.edges.in.flatMap(_.source.sources) ++
        memBlock.inner.debug_int_sink.edges.in.flatMap(_.source.sources) ++
        memBlock.inner.nmi_int_sink.edges.in.flatMap(_.source.sources)
    ).flatMap {
      s =>
        (s.range.start until s.range.`end`).map(_ => s.resources)
    }
    val int_ids       = Seq(
      3,     // msip  [clint]
      7,     // mtip  [clint]
      11,    // meip  [plic]
      9,     // seip  [plic]
      65535, // debug [debug]
      31,    // nmi_31 [nmi]
      43     // nmi_43 [nmi]
    )
    assert(int_resources.size == int_ids.size)
    for ((resources, id) <- int_resources.zip(int_ids)) {
      for (r <- resources) {
        r.bind(intcDevice, ResourceInt(id))
      }
    }
  }
}
