package examples.toy

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile.{HasCoreParameters, LazyRoCC, LazyRoCCModuleImp, OpcodeSet, TileKey}
import freechips.rocketchip.tilelink._

import freechips.rocketchip.tile.{LazyRoCC, LazyRoCCModuleImp}
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.frontend.{Frontend, FrontendParam}
import framework.memdomain.{MemDomain, MemDomainParam}
import framework.gpdomain.{GpDomain, GpDomainParam}
import framework.memdomain.frontend.outside_channel.dma.{BBStreamReader, BBStreamWriter}
import examples.toy.balldomain.{BallDomain, BallDomainParam}
import examples.BuckyballConfigs.CustomBuckyballConfig

class ToyBuckyball(val b: CustomBuckyballConfig)(implicit p: Parameters)
    extends LazyRoCC(opcodes = OpcodeSet.custom3, nPTWPorts = 1) {

  val xLen = p(TileKey).core.xLen // the width of core's register file

  val id_node   = TLIdentityNode()
  val xbar_node = TLXbar()

  val reader = LazyModule(new BBStreamReader(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, b.bankWidth)(p))
  val writer = LazyModule(new BBStreamWriter(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, b.bankWidth)(p))

  // Note: BallDomain is now a regular Module, no longer a LazyModule
  // Will be instantiated in module

  xbar_node := TLBuffer()                        := reader.node
  xbar_node := TLBuffer()                        := writer.node
  id_node   := TLWidthWidget(b.dma_buswidth / 8) := TLBuffer() := xbar_node

  override lazy val module = new ToyBuckyballModule(this)

  // The LazyRoCC class contains two TLOutputNode instances, atlNode and tlNode.
  // atlNode connects into a tile-local arbiter along with the backside of the L1 instruction cache.
  // tlNode connects directly to the L1-L2 crossbar.
  // The corresponding Tilelink ports in the module implementation's IO bundle are atl and tl, respectively.
  override val tlNode  = id_node
  override val atlNode = TLIdentityNode()
  val node             = tlNode
}

class ToyBuckyballModule(outer: ToyBuckyball) extends LazyRoCCModuleImp(outer) with HasCoreParameters {
  import outer.b._

  val tagWidth = 32

// -----------------------------------------------------------------------------
// Frontend: TLB moved inside MemDomain
// -----------------------------------------------------------------------------
  val edge: TLEdgeOut = outer.id_node.edges.out.head

// -----------------------------------------------------------------------------
// Frontend: Global Decoder + Global Reservation Station
// -----------------------------------------------------------------------------
  val b: CustomBuckyballConfig = outer.b

  // Frontend parameter
  val frontendParam = FrontendParam(
    rob_entries = b.rob_entries,
    rs_out_of_order_response = b.rs_out_of_order_response
  )

  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontendParam))
  frontend.io.cmd.valid    := io.cmd.valid
  frontend.io.cmd.bits.cmd := io.cmd.bits
  io.cmd.ready             := frontend.io.cmd.ready

// -----------------------------------------------------------------------------
// Backend: Mem Domain - complete domain containing DMA+TLB+SRAM
// -----------------------------------------------------------------------------
  // Load MemDomain config: try JSON file first, fallback to global config
  val memDomainParam =
    try {
      MemDomainParam.fromJson("arch/src/main/scala/framework/memdomain/configs/default.json")
    } catch {
      case _: Exception => MemDomainParam.fromGlobal(b)
    }

// -----------------------------------------------------------------------------
// Backend: Ball Domain
// -----------------------------------------------------------------------------
  // Load BallDomain config: try JSON file first, fallback to global config
  val ballDomainParam =
    try {
      BallDomainParam.fromJson("arch/src/main/scala/examples/toy/balldomain/configs/default.json")
    } catch {
      case _: Exception => BallDomainParam.fromGlobal(b)
    }

  // Require parameter matching between BallDomain and MemDomain
  require(
    ballDomainParam.bbusChannel == memDomainParam.bankChannel,
    s"BallDomain bbusChannel (${ballDomainParam.bbusChannel}) must equal MemDomain bankChannel (${memDomainParam.bankChannel})"
  )
  require(
    ballDomainParam.numBanks == memDomainParam.bankNum,
    s"BallDomain numBanks (${ballDomainParam.numBanks}) must equal MemDomain bankNum (${memDomainParam.bankNum})"
  )
  require(
    ballDomainParam.bankEntries == memDomainParam.bankEntries,
    s"BallDomain bankEntries (${ballDomainParam.bankEntries}) must equal MemDomain bankEntries (${memDomainParam.bankEntries})"
  )
  require(
    ballDomainParam.bankWidth == memDomainParam.bankWidth,
    s"BallDomain bankWidth (${ballDomainParam.bankWidth}) must equal MemDomain bankWidth (${memDomainParam.bankWidth})"
  )
  require(
    ballDomainParam.bankMaskLen == memDomainParam.bankMaskLen,
    s"BallDomain bankMaskLen (${ballDomainParam.bankMaskLen}) must equal MemDomain bankMaskLen (${memDomainParam.bankMaskLen})"
  )
  require(
    ballDomainParam.rob_entries == memDomainParam.rob_entries,
    s"BallDomain rob_entries (${ballDomainParam.rob_entries}) must equal MemDomain rob_entries (${memDomainParam.rob_entries})"
  )

  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(ballDomainParam)(outer.p))

  // Frontend -> BallDomain
  ballDomain.global_issue_i <> frontend.io.ball_issue_o
  frontend.io.ball_complete_i <> ballDomain.global_complete_o

  val memDomain: Instance[MemDomain] = Instantiate(new MemDomain(memDomainParam)(edge))

  // Frontend -> MemDomain
  memDomain.io.global_issue_i <> frontend.io.mem_issue_o
  frontend.io.mem_complete_i <> memDomain.io.global_complete_o

  // MemDomain -> DMA
  memDomain.io.dma.read.req <> outer.reader.module.io.req
  outer.reader.module.io.resp <> memDomain.io.dma.read.resp
  memDomain.io.dma.write.req <> outer.writer.module.io.req
  outer.writer.module.io.resp <> memDomain.io.dma.write.resp

  // DMA -> TLB (now through MemDomain)
  outer.reader.module.io.tlb <> memDomain.io.tlb(1)
  outer.writer.module.io.tlb <> memDomain.io.tlb(0)

  // PTW connected to MemDomain's TLB (shared TLB has only 1 PTW port)
  io.ptw(0) <> memDomain.io.ptw(0)

  // TLB exception handling - shared TLB has only 1 exception interface
  // Set flush input signals
  memDomain.io.tlbExp(0).flush_skip  := false.B
  memDomain.io.tlbExp(0).flush_retry := false.B

  // Flush signals to DMA components (obtained from MemDomain's TLB exceptions)
  outer.reader.module.io.flush := memDomain.io.tlbExp(0).flush()
  outer.writer.module.io.flush := memDomain.io.tlbExp(0).flush()

// -----------------------------------------------------------------------------
// Backend: Gen Domain - General purpose domain (for T1 vector processor)
// -----------------------------------------------------------------------------
  // Load GpDomain config: try JSON file first, fallback to global config
  val gpDomainParam =
    try {
      GpDomainParam.fromJson("arch/src/main/scala/framework/gpdomain/configs/default.json")
    } catch {
      case _: Exception => GpDomainParam.fromGlobal(b)
    }

  val gpDomain: Instance[GpDomain] = Instantiate(new GpDomain(gpDomainParam)(outer.p))

  // Frontend -> GpDomain
  gpDomain.io.global_issue_i <> frontend.io.gp_issue_o
  frontend.io.gp_complete_i <> gpDomain.io.global_complete_o

// -----------------------------------------------------------------------------
// Backend: Domain Bridge: BallDomain -> MemDomain
// -----------------------------------------------------------------------------
  ballDomain.bankRead <> memDomain.io.ballDomain.bankRead
  ballDomain.bankWrite <> memDomain.io.ballDomain.bankWrite

// ---------------------------------------------------------------------------
// RoCC response and status signals
// ---------------------------------------------------------------------------

  io.resp <> frontend.io.resp
  io.busy      := frontend.io.busy
  io.interrupt := memDomain.io.tlbExp(0).interrupt

// ---------------------------------------------------------------------------
// Busy counter to prevent long simulation stalls
// ---------------------------------------------------------------------------
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(frontend.io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyball: busy for too long!")

}
