package examples.hart.gemmini.native

import chisel3._
import chisel3.util._
import chisel3.experimental.annotate
import sifive.enterprise.firrtl.NestedPrefixModulesAnnotation
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.resources.BindingScope
import freechips.rocketchip.devices.tilelink.{DevNullParams, TLError}
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.tile._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.prci._
import freechips.rocketchip.interrupts._

class Native(implicit p: Parameters) extends LazyModule with BindingScope {
  val params      = p(TilesLocated(InSubsystem)).head.asInstanceOf[RocketTileAttachParams]
  val tile        =
    LazyModule(new RocketTile(params.tileParams, params.crossingParams, HartsWontDeduplicate(params.tileParams)))
  val trace       = tile.traceSourceNode.makeSink()
  val hart        = BundleBridgeSource(() => UInt(1.W))
  val resetVector = BundleBridgeSource(() => UInt(44.W))
  tile.hartIdNode      := hart
  tile.resetVectorNode := resetVector
  val irqs = IntSourceNode(IntSourcePortSimple(5, ports = 1, sources = 1))
  tile.intInwardNode := irqs

  val notifications = Seq(tile.wfiNode, tile.haltNode, tile.ceaseNode).map { source =>
    val sink = IntSinkNode(IntSinkPortSimple(1, 1)); sink := source; sink
  }

  val bus = LazyModule(new TLXbar)

  val error = LazyModule(new TLError(
    DevNullParams(Seq(AddressSet(0x3000, 0xfff)), maxAtomic = 8, maxTransfer = 64),
    beatBytes = 16
  ))

  error.node := bus.node
  bus.node :=* tile.masterNode

  val memory = AXI4SlaveNode(Seq(AXI4SlavePortParameters(
    Seq(AXI4SlaveParameters(
      address = AddressSet(0, (BigInt(1) << 44) - 1).subtract(AddressSet(0, 0x7fffffffL)),
      regionType = RegionType.UNCACHED,
      executable = true,
      supportsRead = TransferSizes(1, 64),
      supportsWrite = TransferSizes(1, 64),
      interleavedId = Some(0)
    )),
    beatBytes = 16
  )))

  memory       := AXI4Buffer()      := AXI4UserYanker()   := AXI4IdIndexer(4) := AXI4Deinterleaver(64) :=
    TLToAXI4() := TLWidthWidget(16) := TLBroadcast(64, 4) := bus.node

  val peripherals = AXI4SlaveNode(Seq(AXI4SlavePortParameters(
    Seq(AXI4SlaveParameters(
      address = AddressSet(0, 0x7fffffffL).subtract(AddressSet(0x3000, 0xfff)),
      regionType = RegionType.UNCACHED,
      executable = true,
      supportsRead = TransferSizes(1, 16),
      supportsWrite = TransferSizes(1, 16),
      interleavedId = Some(0)
    )),
    beatBytes = 16
  )))

  // Split DMA-sized requests into the single-beat accesses accepted by device peripherals.
  peripherals  := AXI4Buffer()                               := AXI4UserYanker()  := AXI4IdIndexer(4) := AXI4Deinterleaver(64) :=
    TLToAXI4() := TLFragmenter(16, 64, holdFirstDeny = true) := TLWidthWidget(16) := bus.node

  lazy val module = new LazyModuleImp(this) {
    annotate(this)(Seq(NestedPrefixModulesAnnotation(this.toNamed, "GemminiHart", true)))

    val io = IO(new Bundle {
      val resetVector = Input(UInt(44.W))
      val interrupts  = Input(Vec(4, Bool()))
    })

    val memoryAXI      = memory.makeIOs()
    val peripheralsAXI = peripherals.makeIOs()
    hart.bundle        := 0.U
    resetVector.bundle := io.resetVector
    irqs.out.head._1   := VecInit(false.B +: io.interrupts.toSeq)
  }

}

object Emit extends App {

  implicit val p: Parameters =
    new Config((site, here, up) => {
      case TilesLocated(InSubsystem) => up(TilesLocated(InSubsystem)).map {
          case tile: RocketTileAttachParams =>
            tile.copy(tileParams = tile.tileParams.copy(core = tile.tileParams.core.copy(pgLevels = 4)))
        }
      case MaxHartIdBits             => 1
      case SystemBusKey              => up(SystemBusKey).copy(beatBytes = 16)
    }) ++ new _root_.gemmini.DefaultGemminiConfig ++ new freechips.rocketchip.rocket.WithNBigCores(
      1
    ) ++ new BaseSubsystemConfig

  val native = LazyModule(new Native)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    native.module,
    args ++ Array("--split-verilog"),
    firtoolOpts = Array("--disable-all-randomization")
  )
}
