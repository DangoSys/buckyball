package examples.toy

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._

import freechips.rocketchip.tile.{LazyRoCC, LazyRoCCModuleImp}
import framework.frontend.decoder.GlobalDecoder
import framework.memdomain.dma.{BBStreamReader, BBStreamWriter}
import framework.memdomain.MemDomain
import examples.toy.balldomain.BallDomain
import examples.BuckyballConfigs.CustomBuckyballConfig


class ToyBuckyball(val b: CustomBuckyballConfig)(implicit p: Parameters)
  extends LazyRoCC (opcodes = b.opcodes, nPTWPorts = 1) {

  val xLen = p(TileKey).core.xLen   // the width of core's register file

  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()

  val spad_w = b.inputType.getWidth * b.veclane
  val reader = LazyModule(new BBStreamReader(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))
  val writer = LazyModule(new BBStreamWriter(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))

  // Note: BallDomain is now a regular Module, no longer a LazyModule
  // Will be instantiated in module

  xbar_node := TLBuffer() := reader.node
  xbar_node := TLBuffer() := writer.node
  id_node := TLWidthWidget(b.dma_buswidth/8) := TLBuffer() := xbar_node

  override lazy val module = new ToyBuckyballModule(this)

  // The LazyRoCC class contains two TLOutputNode instances, atlNode and tlNode.
  // atlNode connects into a tile-local arbiter along with the backside of the L1 instruction cache.
  // tlNode connects directly to the L1-L2 crossbar.
  // The corresponding Tilelink ports in the module implementation's IO bundle are atl and tl, respectively.
  override val tlNode = id_node
  override val atlNode = TLIdentityNode()
  val node = tlNode
}

class ToyBuckyballModule(outer: ToyBuckyball) extends LazyRoCCModuleImp(outer)
  with HasCoreParameters {
  import outer.b._

  val tagWidth = 32

// -----------------------------------------------------------------------------
// Frontend: TLB moved inside MemDomain
// -----------------------------------------------------------------------------
  implicit val edge: TLEdgeOut = outer.id_node.edges.out.head

// -----------------------------------------------------------------------------
// Frontend: Global Decoder + Global Reservation Station
// -----------------------------------------------------------------------------
  implicit val b: CustomBuckyballConfig = outer.b

  val gDecoder = Module(new GlobalDecoder)
  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready              := gDecoder.io.id_i.ready

  // Global reservation station
  val globalRs = Module(new framework.frontend.globalrs.GlobalReservationStation)
  globalRs.io.global_decode_cmd_i <> gDecoder.io.id_o



// -----------------------------------------------------------------------------
// Backend: Ball Domain
// -----------------------------------------------------------------------------
  // BallDomain is now a regular Module, instantiate directly
  val ballDomain = Module(new BallDomain()(b, p))

  // Global RS -> BallDomain
  ballDomain.io.global_issue_i <> globalRs.io.ball_issue_o
  globalRs.io.ball_complete_i <> ballDomain.io.global_complete_o

// -----------------------------------------------------------------------------
// Backend: Mem Domain - complete domain containing DMA+TLB+SRAM
// -----------------------------------------------------------------------------
  val memDomain = Module(new MemDomain)

  // Global RS -> MemDomain
  memDomain.io.global_issue_i <> globalRs.io.mem_issue_o
  globalRs.io.mem_complete_i <> memDomain.io.global_complete_o

// -----------------------------------------------------------------------------
// Backend: MemDomain Connections
// -----------------------------------------------------------------------------
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
  memDomain.io.tlbExp(0).flush_skip := false.B
  memDomain.io.tlbExp(0).flush_retry := false.B

  // Flush signals to DMA components (obtained from MemDomain's TLB exceptions)
  outer.reader.module.io.flush := memDomain.io.tlbExp(0).flush()
  outer.writer.module.io.flush := memDomain.io.tlbExp(0).flush()

// -----------------------------------------------------------------------------
// Backend: Domain Bridge: BallDomain -> MemDomain
// -----------------------------------------------------------------------------
  ballDomain.io.sramRead  <> memDomain.io.ballDomain.sramRead
  ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite

// ---------------------------------------------------------------------------
// RoCC response and status signals
// ---------------------------------------------------------------------------
  io.resp <> globalRs.io.rs_rocc_o.resp
  io.busy := globalRs.io.rs_rocc_o.busy
  io.interrupt := memDomain.io.tlbExp(0).interrupt

// ---------------------------------------------------------------------------
// Busy counter to prevent long simulation stalls
// ---------------------------------------------------------------------------
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(globalRs.io.rs_rocc_o.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyball: busy for too long!")

}
