package examples.toy

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._

import framework.rocket.{LazyRoCCBB, LazyRoCCModuleImpBB, RoCCResponseBB}
import framework.builtin.frontend.GlobalDecoder
import framework.builtin.memdomain.dma.{BBStreamReader, BBStreamWriter}
import framework.builtin.memdomain.MemDomain
import examples.toy.balldomain.BallDomain
import examples.BuckyBallConfigs.CustomBuckyBallConfig


class ToyBuckyBall(val b: CustomBuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = b.opcodes, nPTWPorts = 2) {

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

  override lazy val module = new ToyBuckyBallModule(this)

  // The LazyRoCC class contains two TLOutputNode instances, atlNode and tlNode.
  // atlNode connects into a tile-local arbiter along with the backside of the L1 instruction cache.
  // tlNode connects directly to the L1-L2 crossbar.
  // The corresponding Tilelink ports in the module implementation's IO bundle are atl and tl, respectively.
  override val tlNode = id_node
  override val atlNode = TLIdentityNode()
  val node = tlNode
}

class ToyBuckyBallModule(outer: ToyBuckyBall) extends LazyRoCCModuleImpBB(outer)
  with HasCoreParameters {
  import outer.b._

  val tagWidth = 32

// -----------------------------------------------------------------------------
// Frontend: TLB moved inside MemDomain
// -----------------------------------------------------------------------------
  implicit val edge: TLEdgeOut = outer.id_node.edges.out.head

// -----------------------------------------------------------------------------
// Frontend: Global Decoder -> Global RS -> BallDomain/MemDomain
// -----------------------------------------------------------------------------
  implicit val b: CustomBuckyBallConfig = outer.b
  val gDecoder = Module(new GlobalDecoder)
  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready              := gDecoder.io.id_i.ready

  // Global reservation station
  val globalRs = Module(new framework.builtin.frontend.globalrs.GlobalReservationStation)
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

  // PTW connected to MemDomain's TLB
  io.ptw <> memDomain.io.ptw

  // TLB exception handling - MemDomain's tlbExp is now Output, so we read signals from it
  // Set flush input signals
  memDomain.io.tlbExp.foreach { exp =>
    exp.flush_skip := false.B
    exp.flush_retry := false.B
  }

  // Flush signals to DMA components (obtained from MemDomain's TLB exceptions)
  outer.reader.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)
  outer.writer.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)

// -----------------------------------------------------------------------------
// Backend: Domain Bridge: BallDomain -> MemDomain
// -----------------------------------------------------------------------------
  ballDomain.io.sramRead  <> memDomain.io.ballDomain.sramRead
  ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
  ballDomain.io.accRead   <> memDomain.io.ballDomain.accRead
  ballDomain.io.accWrite  <> memDomain.io.ballDomain.accWrite

// ---------------------------------------------------------------------------
// Return RoCC interface connection - get response from global RS
// ---------------------------------------------------------------------------
  io.resp <> globalRs.io.rs_rocc_o.resp


// ---------------------------------------------------------------------------
// Busy signal - managed by global RS
// ---------------------------------------------------------------------------
  //io.busy := globalRs.io.rs_rocc_o.busy
  io.busy := false.B

// ---------------------------------------------------------------------------
// Busy counter to prevent long simulation stalls
// ---------------------------------------------------------------------------
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(globalRs.io.rs_rocc_o.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyBall: busy for too long!")

}
