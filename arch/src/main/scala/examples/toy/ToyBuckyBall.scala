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

  // 注意：BallDomain现在是普通Module，不再是LazyModule
  // 将在module中实例化

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
// Frontend: TLB 移至MemDomain内部
// -----------------------------------------------------------------------------
  implicit val edge: TLEdgeOut = outer.id_node.edges.out.head

// -----------------------------------------------------------------------------
// Frontend: Global Decoder -> 全局RS -> BallDomain/MemDomain
// -----------------------------------------------------------------------------
  implicit val b: CustomBuckyBallConfig = outer.b
  val gDecoder = Module(new GlobalDecoder)
  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready              := gDecoder.io.id_i.ready

  // 全局保留站
  val globalRs = Module(new framework.builtin.frontend.globalrs.GlobalReservationStation)
  globalRs.io.global_decode_cmd_i <> gDecoder.io.id_o



// -----------------------------------------------------------------------------
// Backend: Ball Domain
// -----------------------------------------------------------------------------
  // BallDomain现在是普通Module，直接实例化
  val ballDomain = Module(new BallDomain()(b, p))

  // 全局RS->BallDomain
  ballDomain.io.global_issue_i <> globalRs.io.ball_issue_o
  globalRs.io.ball_complete_i <> ballDomain.io.global_complete_o

// -----------------------------------------------------------------------------
// Backend: Mem Domain 包含DMA+TLB+SRAM的完整域
// -----------------------------------------------------------------------------
  val memDomain = Module(new MemDomain)

  // 全局RS->MemDomain
  memDomain.io.global_issue_i <> globalRs.io.mem_issue_o
  globalRs.io.mem_complete_i <> memDomain.io.global_complete_o

// -----------------------------------------------------------------------------
// Backend: MemDomain Connections
// -----------------------------------------------------------------------------
  // MemDomain->DMA
  memDomain.io.dma.read.req <> outer.reader.module.io.req
  outer.reader.module.io.resp <> memDomain.io.dma.read.resp
  memDomain.io.dma.write.req <> outer.writer.module.io.req
  outer.writer.module.io.resp <> memDomain.io.dma.write.resp

  // DMA->TLB (现在通过MemDomain)
  outer.reader.module.io.tlb <> memDomain.io.tlb(1)
  outer.writer.module.io.tlb <> memDomain.io.tlb(0)

  // PTW连接到MemDomain的TLB
  io.ptw <> memDomain.io.ptw

  // TLB异常处理 - 现在MemDomain的tlbExp是Output，所以我们从它读取信号
  // 设置flush输入信号
  memDomain.io.tlbExp.foreach { exp =>
    exp.flush_skip := false.B
    exp.flush_retry := false.B
  }

  // Flush信号给DMA组件 (从MemDomain的TLB异常获取)
  outer.reader.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)
  outer.writer.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)

// -----------------------------------------------------------------------------
// Backend: Domain Bridge: BallDomain->MemDomain
// -----------------------------------------------------------------------------
  ballDomain.io.sramRead  <> memDomain.io.ballDomain.sramRead
  ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
  ballDomain.io.accRead   <> memDomain.io.ballDomain.accRead
  ballDomain.io.accWrite  <> memDomain.io.ballDomain.accWrite

// ---------------------------------------------------------------------------
// 返回RoCC接口连接 - 从全局RS获取响应
// ---------------------------------------------------------------------------
  io.resp <> globalRs.io.rs_rocc_o.resp


// ---------------------------------------------------------------------------
// Busy信号 - 由全局RS管理
// ---------------------------------------------------------------------------
  io.busy := globalRs.io.rs_rocc_o.busy

// ---------------------------------------------------------------------------
// busy计数器，防止仿真长时间停顿
// ---------------------------------------------------------------------------
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "ToyBuckyBall: busy for too long!")

}
