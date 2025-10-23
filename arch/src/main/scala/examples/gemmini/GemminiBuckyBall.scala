package examples.gemmini

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._

import framework.rocket.{LazyRoCCBB, LazyRoCCModuleImpBB, RoCCResponseBB}
import framework.builtin.memdomain.dma.{BBStreamReader, BBStreamWriter}
import framework.builtin.memdomain.MemDomain
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.gemmini.decoder.GemminiDecoder
import examples.gemmini.balldomain.GemminiBallDomain

/**
 * GemminiBuckyBall: Gemmini加速器的BuckyBall实现
 * 架构类似ToyBuckyBall，但添加了GemminiDecoder来转译指令
 */
class GemminiBuckyBall(val b: CustomBuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = b.opcodes, nPTWPorts = 2) {

  val xLen = p(TileKey).core.xLen

  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()

  val spad_w = b.inputType.getWidth * b.veclane
  val reader = LazyModule(new BBStreamReader(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))
  val writer = LazyModule(new BBStreamWriter(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))

  xbar_node := TLBuffer() := reader.node
  xbar_node := TLBuffer() := writer.node
  id_node := TLWidthWidget(b.dma_buswidth/8) := TLBuffer() := xbar_node

  override lazy val module = new GemminiBuckyBallModule(this)

  override val tlNode = id_node
  override val atlNode = TLIdentityNode()
  val node = tlNode
}

class GemminiBuckyBallModule(outer: GemminiBuckyBall) extends LazyRoCCModuleImpBB(outer)
  with HasCoreParameters {
  import outer.b._

  val tagWidth = 32

  implicit val edge: TLEdgeOut = outer.id_node.edges.out.head
  implicit val b: CustomBuckyBallConfig = outer.b

  // Gemmini指令解码器
  val gemminiDecoder = Module(new GemminiDecoder)
  gemminiDecoder.io.gDecoderIn.valid := io.cmd.valid
  gemminiDecoder.io.gDecoderIn.bits.is_ball := false.B  // 先设为false，由decoder判断
  gemminiDecoder.io.gDecoderIn.bits.is_mem := false.B
  gemminiDecoder.io.gDecoderIn.bits.raw_cmd := io.cmd.bits
  io.cmd.ready := gemminiDecoder.io.gDecoderIn.ready

  // Ball Domain (包含Gemmini的计算Ball)
  val ballDomain = Module(new GemminiBallDomain()(b, p))
  ballDomain.io.gDecoderIn <> gemminiDecoder.io.ballDecoderOut

  // Mem Domain (处理mvin/mvout)
  val memDomain = Module(new MemDomain)
  memDomain.io.gDecoderIn <> gemminiDecoder.io.memDecoderOut

  // DMA连接
  memDomain.io.dma.read.req <> outer.reader.module.io.req
  outer.reader.module.io.resp <> memDomain.io.dma.read.resp
  memDomain.io.dma.write.req <> outer.writer.module.io.req
  outer.writer.module.io.resp <> memDomain.io.dma.write.resp

  // TLB连接
  outer.reader.module.io.tlb <> memDomain.io.tlb(1)
  outer.writer.module.io.tlb <> memDomain.io.tlb(0)
  io.ptw <> memDomain.io.ptw

  // TLB异常处理
  memDomain.io.tlbExp.foreach { exp =>
    exp.flush_skip := false.B
    exp.flush_retry := false.B
  }
  outer.reader.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)
  outer.writer.module.io.flush := memDomain.io.tlbExp.map(_.flush()).reduce(_ || _)

  // Domain Bridge: BallDomain <-> MemDomain
  ballDomain.io.sramRead <> memDomain.io.ballDomain.sramRead
  ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
  ballDomain.io.accRead <> memDomain.io.ballDomain.accRead
  ballDomain.io.accWrite <> memDomain.io.ballDomain.accWrite

  // RoCC响应仲裁
  val respArb = Module(new Arbiter(new RoCCResponseBB()(p), 2))
  respArb.io.in(0) <> ballDomain.io.roccResp
  respArb.io.in(1) <> memDomain.io.roccResp
  io.resp <> respArb.io.out

  // Busy信号
  io.busy := ballDomain.io.busy || memDomain.io.busy

  // 防止仿真长时间停顿
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "GemminiBuckyBall: busy for too long!")

  override lazy val desiredName = "GemminiBuckyBall"
}
