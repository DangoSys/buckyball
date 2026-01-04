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
import framework.top.GlobalConfig
import examples.toy.balldomain.BallDomain
import framework.frontend.Frontend
import framework.gpdomain.GpDomain
import framework.memdomain.MemDomain
import framework.top.channels.{Channel, ChannelCluster, ChannelIO}

class ToyBuckyball(val b: GlobalConfig)(implicit p: Parameters)
    extends LazyRoCC(opcodes = OpcodeSet.custom3, nPTWPorts = 1) {

  // the width of core's register file
  val xLen = p(TileKey).core.xLen

  // DMA nodes for read and write paths
  val reader_node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "bb-dma-reader",
    sourceId = freechips.rocketchip.diplomacy.IdRange(0, b.memDomain.dma_n_xacts)
  )))))

  val writer_node = TLClientNode(Seq(TLMasterPortParameters.v1(Seq(TLClientParameters(
    name = "bb-dma-writer",
    sourceId = freechips.rocketchip.diplomacy.IdRange(0, b.memDomain.dma_n_xacts)
  )))))

  val xbar_node = TLXbar()

  // Connect reader and writer to xbar
  xbar_node := TLBuffer() := reader_node
  xbar_node := TLBuffer() := writer_node

  override lazy val module = new ToyBuckyballModule(this)

  // tlNode connects to mbus (bypassing L2 cache)
  override val tlNode = TLWidthWidget(b.memDomain.dma_buswidth / 8) := TLBuffer() := xbar_node

  // atlNode is not used (set to identity for compatibility)
  override val atlNode = TLIdentityNode()
}

class ToyBuckyballModule(outer: ToyBuckyball) extends LazyRoCCModuleImp(outer) with HasCoreParameters {
  import outer.b._
  val b: GlobalConfig = outer.b

  // Get TileLink edges from reader and writer nodes
  val (tl_reader, edge_reader) = outer.reader_node.out(0)
  val (tl_writer, edge_writer) = outer.writer_node.out(0)

  val frontend:              Instance[Frontend]       = Instantiate(new Frontend(b))
  val ballDomain:            Instance[BallDomain]     = Instantiate(new BallDomain(b))
  val memDomain:             Instance[MemDomain]      = Instantiate(new MemDomain(b)(edge_reader))
  val gpDomain:              Instance[GpDomain]       = Instantiate(new GpDomain(b))
  val ballMemChannelCluster: Instance[ChannelCluster] = Instantiate(new ChannelCluster(b, b.top.ballMemChannelNum))
  val memBallChannelCluster: Instance[ChannelCluster] = Instantiate(new ChannelCluster(b, b.top.memBallChannelNum))

  frontend.io.cmd.valid    := io.cmd.valid
  frontend.io.cmd.bits.cmd := io.cmd.bits
  io.cmd.ready             := frontend.io.cmd.ready

  // Frontend -> BallDomain
  ballDomain.global_issue_i <> frontend.io.ball_issue_o
  frontend.io.ball_complete_i <> ballDomain.global_complete_o

  // Frontend -> MemDomain
  memDomain.io.global_issue_i <> frontend.io.mem_issue_o
  frontend.io.mem_complete_i <> memDomain.io.global_complete_o

  // PTW connected to MemDomain's TLB (shared TLB has only 1 PTW port)
  // Connect all fields except customCSRs which is not supported in our BBTLBPTWIO
  io.ptw(0).req <> memDomain.io.ptw(0).req
  memDomain.io.ptw(0).resp <> io.ptw(0).resp
  memDomain.io.ptw(0).ptbr <> io.ptw(0).ptbr
  memDomain.io.ptw(0).hgatp <> io.ptw(0).hgatp
  memDomain.io.ptw(0).vsatp <> io.ptw(0).vsatp
  memDomain.io.ptw(0).status <> io.ptw(0).status
  memDomain.io.ptw(0).hstatus <> io.ptw(0).hstatus
  memDomain.io.ptw(0).gstatus <> io.ptw(0).gstatus
  memDomain.io.ptw(0).pmp <> io.ptw(0).pmp
  // customCSRs is tied off - we don't use custom CSRs in our implementation
  memDomain.io.ptw(0).customCSRs := DontCare

  // TLB exception handling - shared TLB has only 1 exception interface
  // Set flush input signals
  memDomain.io.tlbExp(0).flush_skip  := false.B
  memDomain.io.tlbExp(0).flush_retry := false.B

  // Frontend -> GpDomain
  gpDomain.io.global_issue_i <> frontend.io.gp_issue_o
  frontend.io.gp_complete_i <> gpDomain.io.global_complete_o

  ballDomain.bankRead <> memDomain.io.ballDomain.bankRead
  ballDomain.bankWrite <> memDomain.io.ballDomain.bankWrite

  // Connect BallMemChannelCluster
  ballMemChannelCluster.io <> ballDomain.ballMemChannel

  // Connect MemBallChannelCluster
  memBallChannelCluster.io <> ballDomain.memBallChannel

  // Connect TileLink DMA ports from MemDomain to LazyModule nodes
  tl_reader <> memDomain.io.tl_reader
  tl_writer <> memDomain.io.tl_writer

  io.resp <> frontend.io.resp
  io.busy      := frontend.io.busy
  io.interrupt := memDomain.io.tlbExp(0).interrupt

  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(frontend.io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyball: busy for too long!")

}
