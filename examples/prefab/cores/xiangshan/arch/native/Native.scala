package examples.core.xiangshan.native

import chisel3._
import chisel3.experimental.annotate
import sifive.enterprise.firrtl.NestedPrefixModulesAnnotation
import chisel3.util._
import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.interrupts._
import freechips.rocketchip.tilelink._
import coupledL2.{EnableCHI, L2ParamKey}
import coupledL2.tl2chi.{CHIAddrWidthKey, CHIIssue, PortIO}
import system.SoCParamsKey
import top.{ArgParser, Generator, SeperatedBusType}
import utility.{ChiselDB, Constantin}
import freechips.rocketchip.util.uintToBitPat
import xiangshan._
import xiangshan.backend.decode.XSDecode
import xiangshan.backend.fu.FuType
import examples.core.xiangshan.native.XSTile

class Context(implicit p: Parameters) extends XSBundle with HasPMParameters {
  val satp      = UInt(64.W)
  val privilege = UInt(2.W)
  val sum       = Bool()
  val mxr       = Bool()
  val virt      = Bool()

  val pmp = Vec(
    NumPMP,
    new Bundle {
      val cfg  = UInt(8.W)
      val addr = UInt((PMPAddrBits - 2).W)
    }
  )

}

/** Accelerator command: custom-3 with funct3 = 3 and rd = 0, executed by the CSR unit. */
object Command {
  def opType = 0.U(FuOpTypeWidth.W)

  def decode = BitPat("b???????_?????_?????_011_00000_1111011") ->
    XSDecode(
      SrcType.reg,
      SrcType.reg,
      SrcType.X,
      FuType.csr,
      opType,
      SelImm.IMM_Z,
      xWen = true,
      noSpec = true,
      blockBack = true
    )
      .generate()

}

class Commands(implicit p: Parameters) extends XSBundle {

  val request = Decoupled(new Bundle {
    val instruction = UInt(32.W)
    val pc          = UInt(64.W)
    val rs1         = UInt(64.W)
    val rs2         = UInt(64.W)
    val context     = new Context
  })

  val complete = Flipped(Valid(Bool()))
}

/** Native XiangShan tile boundary; its caches retain the upstream CHI endpoint. */
class Native(implicit p: Parameters) extends LazyModule with BindingScope {
  val tile        = LazyModule(new XSTile)
  val resetSource = BundleBridgeSource(() => Reset())
  tile.core_reset_sink := resetSource
  val clint    = IntSourceNode(IntSourcePortSimple(2, ports = 1, sources = 1))
  val plic     = IntSourceNode(IntSourcePortSimple(1, ports = 2, sources = 1))
  val debug    = IntSourceNode(IntSourcePortSimple(1, ports = 1, sources = 1))
  val nmi      = IntSourceNode(IntSourcePortSimple(2, ports = 1, sources = 1))
  val busError = IntSinkNode(IntSinkPortSimple(1, 1))
  tile.clint_int_node := clint
  tile.plic_int_node :*= plic
  tile.debug_int_node := debug
  tile.nmi_int_node   := nmi
  busError            := tile.beu_int_source

  val uncached = AXI4SlaveNode(Seq(AXI4SlavePortParameters(
    slaves = Seq(AXI4SlaveParameters(
      address = AddressSet(0, 0x7fffffffL).subtract(p(SoCParamsKey).BEURange),
      regionType = RegionType.UNCACHED,
      executable = true,
      supportsRead = TransferSizes(1, 16),
      supportsWrite = TransferSizes(1, 16),
      interleavedId = Some(0)
    )),
    beatBytes = 16
  )))

  uncached                := AXI4Buffer() := AXI4UserYanker() := AXI4IdIndexer(4) :=
    AXI4Deinterleaver(64) := TLToAXI4()   := TLWidthWidget(8) := tile.sep_tl_opt.get

  lazy val module = new LazyModuleImp(this) with RequireAsyncReset {
    annotate(this)(Seq(NestedPrefixModulesAnnotation(this.toNamed, "XiangShan", true)))

    val io = IO(new Bundle {
      val hartId      = Input(UInt(p(freechips.rocketchip.tile.MaxHartIdBits).W))
      val resetVector = Input(UInt(p(SoCParamsKey).PAddrBits.W))
      val nodeId      = Input(UInt(7.W))
      val interrupts  = Input(Vec(4, Bool()))
      val commands    = new Commands
      val timeRequest = Decoupled(Bool())
      val time        = Flipped(Valid(UInt(64.W)))
      val chi         = new PortIO
    })

    val uncachedAXI = uncached.makeIOs()
    io.chi <> tile.module.io.chi.get
    tile.module.io.nodeID.get                            := io.nodeId
    tile.module.io.hartId                                := io.hartId
    tile.module.io.reset_vector                          := io.resetVector
    tile.module.io.reset_mtvec.foreach(_                 := io.resetVector)
    tile.module.io.msiInfo.valid                         := false.B
    tile.module.io.msiInfo.bits                          := 0.U
    tile.module.io.teemsiInfo.foreach { x => x.valid := false.B; x.bits := 0.U }
    tile.module.io.traceCoreInterface.fromEncoder.enable := false.B
    tile.module.io.traceCoreInterface.fromEncoder.stall  := false.B
    tile.module.io.debugTopDown.l3MissMatch              := false.B
    tile.module.io.l3Miss                                := false.B
    tile.module.io.clintTime <> io.time
    io.commands <> tile.module.io.commands
    io.timeRequest <> tile.module.io.timeRequest
    resetSource.out.head._1                              := false.B.asAsyncReset
    clint.out.head._1(0)                                 := io.interrupts(0)
    clint.out.head._1(1)                                 := io.interrupts(1)
    plic.out(0)._1(0)                                    := io.interrupts(2) || busError.in.head._1(0)
    plic.out(1)._1(0)                                    := io.interrupts(3)
    debug.out.head._1(0)                                 := false.B
    nmi.out.head._1.foreach(_                            := false.B)
  }

}

object Emit extends App {

  val (base, stageArgs, upstreamOptions) = ArgParser.parse(
    Array("--config", "KunminghuV2Config", "--fpga-platform", "--disable-perf", "--disable-xmr") ++ args
  )

  val profile = base.alter((site, _, up) => {
    case EnableCHI       => true
    case CHIIssue        => "B"
    case CHIAddrWidthKey => 48
    case SoCParamsKey    => up(SoCParamsKey).copy(
        PAddrBits = 48,
        EnableICacheCtrl = false,
        EnableDCacheCtrl = false,
        SeperateBus = SeperatedBusType.TL,
        SeperateBusRanges = Seq(AddressSet(0, 0x7fffffffL))
      )
    case DFTOptionsKey   => up(DFTOptionsKey).copy(EnableMbist = false, EnableSramCtl = false)
    case DebugOptionsKey => up(DebugOptionsKey).copy(
        FPGAPlatform = true,
        EnableDifftest = false,
        AlwaysBasicDiff = false,
        EnableDebug = false,
        EnablePerfDebug = false,
        EnableChiselDB = false,
        AlwaysBasicDB = false
      )
  })

  val selected = profile(XSTileKey).head

  val core = selected.copy(L2CacheParamsOpt =
    selected.L2CacheParamsOpt.map(
      _.copy(dataCheck = None, enablePoison = false)
    )
  )

  val soc = profile(SoCParamsKey)

  val config = profile.alterPartial {
    case XSCoreParamsKey => core
    case SoCParamsKey    => soc
    case L2ParamKey      => core.L2CacheParamsOpt.get
  }

  Constantin.init(false)
  ChiselDB.init(false)
  val native = LazyModule(new Native()(config))
  val output = args.sliding(2).collectFirst { case Array("--target-dir", path) => path }.get
  Generator.execute(
    stageArgs,
    native.module,
    upstreamOptions ++ Array(
      "--split-verilog",
      s"-o=$output",
      "--disable-annotation-unknown",
      "--lowering-options=explicitBitcast,disallowLocalVariables,disallowPortDeclSharing"
    )
  )
}
