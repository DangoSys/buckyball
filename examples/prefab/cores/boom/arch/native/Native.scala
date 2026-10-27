package examples.core.boom.native

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
import boom.v4.common._
import examples.core.boom.native.BoomTile

class Context extends Bundle {
  val satp = UInt(64.W)
  val pmp  = Vec(8, new Bundle { val cfg = UInt(8.W); val addr = UInt(42.W) })
}

class Commands extends Bundle {

  val request = Decoupled(new Bundle {
    val instruction = UInt(32.W)
    val rs1         = UInt(64.W)
    val rs2         = UInt(64.W)
    val privilege   = UInt(2.W)
    val sum         = Bool()
    val mxr         = Bool()
  })

  val busy  = Input(Bool())
  val fault = Input(Bool())
}

class Accelerator(implicit p: Parameters) extends LazyRoCC(OpcodeSet.custom3) {
  val command = BundleBridgeSource(() => new Commands)

  override lazy val module = new LazyRoCCModuleImp(this) {
    val c = command.bundle
    c.request.valid            := io.cmd.valid
    c.request.bits.instruction := io.cmd.bits.inst.asUInt
    c.request.bits.rs1         := io.cmd.bits.rs1
    c.request.bits.rs2         := io.cmd.bits.rs2
    c.request.bits.privilege   := io.cmd.bits.status.dprv
    c.request.bits.sum         := io.cmd.bits.status.sum
    c.request.bits.mxr         := io.cmd.bits.status.mxr
    io.cmd.ready               := c.request.ready
    when(io.cmd.fire) { assert(!io.cmd.bits.inst.xd); assert(!io.cmd.bits.status.dv) }
    io.busy                    := c.busy
    io.interrupt               := c.fault
    io.resp.valid              := false.B
    io.resp.bits               := 0.U.asTypeOf(io.resp.bits)
    io.mem.req.valid           := false.B
    io.mem.req.bits            := 0.U.asTypeOf(io.mem.req.bits)
    io.mem.s1_kill             := true.B
    io.mem.s2_kill             := true.B
    io.mem.s1_data             := 0.U.asTypeOf(io.mem.s1_data)
    io.mem.keep_clock_enabled  := false.B
    io.fpu_req.valid           := false.B
    io.fpu_req.bits            := 0.U.asTypeOf(io.fpu_req.bits)
    io.fpu_resp.ready          := true.B
  }

}

class Native(implicit p: Parameters) extends LazyModule with BindingScope {
  val params      = p(TilesLocated(InSubsystem)).head.asInstanceOf[BoomTileAttachParams]
  val tile        = LazyModule(new BoomTile(params.tileParams, params.crossingParams, params.lookup))
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

  val commands = BundleBridgeSink[Commands]()
  commands := tile.roccs.head.asInstanceOf[Accelerator].command
  val context = BundleBridgeSink[Context]()
  context := tile.buckyballContext

  val dma = AXI4MasterNode(Seq(AXI4MasterPortParameters(Seq(AXI4MasterParameters(
    name = "accelerator",
    id = IdRange(0, 16),
    aligned = true,
    maxFlight = Some(4)
  )))))

  val bus = LazyModule(new TLXbar)

  val error = LazyModule(new TLError(
    DevNullParams(Seq(AddressSet(0x3000, 0xfff)), maxAtomic = 8, maxTransfer = 64),
    beatBytes = 16
  ))

  error.node := bus.node
  bus.node   := tile.masterNode
  bus.node   := TLFIFOFixer(TLFIFOFixer.all) := AXI4ToTL() := AXI4UserYanker(Some(4)) := AXI4Fragmenter() := dma

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

  peripherals  := AXI4Buffer()      := AXI4UserYanker() := AXI4IdIndexer(4) := AXI4Deinterleaver(64) :=
    TLToAXI4() := TLWidthWidget(16) := bus.node

  lazy val module = new LazyModuleImp(this) {
    annotate(this)(Seq(NestedPrefixModulesAnnotation(this.toNamed, "Boom", true)))

    val io = IO(new Bundle {
      val resetVector = Input(UInt(44.W))
      val interrupts  = Input(Vec(4, Bool()))
      val commands    = new Commands
      val context     = Output(new Context)
    })

    val memoryAXI      = memory.makeIOs()
    val peripheralsAXI = peripherals.makeIOs()
    val dmaAXI         = dma.makeIOs()
    hart.bundle        := 0.U
    resetVector.bundle := io.resetVector
    irqs.out.head._1   := VecInit(false.B +: io.interrupts.toSeq)
    io.commands <> commands.bundle
    io.context         := context.bundle
  }

}

object Emit extends App {

  implicit val p: Parameters =
    new Config((site, here, up) => {
      case TilesLocated(InSubsystem) => up(TilesLocated(InSubsystem)).map {
          case tile: BoomTileAttachParams =>
            tile.copy(tileParams = tile.tileParams.copy(core = tile.tileParams.core.copy(pgLevels = 4)))
        }
      case BuildRoCC                 => Seq((p: Parameters) => LazyModule(new Accelerator()(p)))
      case MaxHartIdBits             => 1
    }) ++ new WithNSmallBooms(1) ++ new BaseSubsystemConfig

  val native = LazyModule(new Native)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    native.module,
    args ++ Array("--split-verilog"),
    firtoolOpts = Array("--disable-all-randomization")
  )
}
