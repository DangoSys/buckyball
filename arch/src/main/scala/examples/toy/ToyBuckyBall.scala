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
import framework.builtin.frontend.{FrontendTLB, GlobalDecoder}
import framework.builtin.memdomain.dma.{BBStreamReader, BBStreamWriter}
import framework.builtin.memdomain.MemDomain
import examples.toy.balldomain.BallDomain
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.FenceCSR 


class ToyBuckyBall(val b: CustomBuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = b.opcodes, nPTWPorts = 2) {

  val xLen = p(TileKey).core.xLen   // the width of core's register file
  
  // DMA组件现在在BuckyBall层面, 后面移到MemDomain内部
  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()
  
  val spad_w = b.inputType.getWidth * b.veclane
  val reader = LazyModule(new BBStreamReader(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))
  val writer = LazyModule(new BBStreamWriter(b.max_in_flight_mem_reqs, b.dma_buswidth, b.dma_maxbytes, spad_w))

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
// Frontend: TLB
// -----------------------------------------------------------------------------
  implicit val edge: TLEdgeOut = outer.id_node.edges.out.head
  val tlb = Module(new FrontendTLB(2, tlb_size, dma_maxbytes))

  tlb.io.exp.foreach(_.flush_skip := false.B)
  tlb.io.exp.foreach(_.flush_retry := false.B)

  io.ptw <> tlb.io.ptw

  // Flush信号给DMA组件
  outer.reader.module.io.flush := tlb.io.exp.map(_.flush()).reduce(_ || _)
  outer.writer.module.io.flush := tlb.io.exp.map(_.flush()).reduce(_ || _)

// -----------------------------------------------------------------------------
// Frontend: Global Decode Dispatch commands to BallDomain and MemDomain
// -----------------------------------------------------------------------------
  implicit val b: CustomBuckyBallConfig = outer.b
  val gDecoder = Module(new GlobalDecoder)
  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready              := gDecoder.io.id_i.ready



// -----------------------------------------------------------------------------
// Backend: Ball Domain
// -----------------------------------------------------------------------------
  val ballDomain = Module(new BallDomain)
  
  // GlobalDecoder->BallDomain 
  ballDomain.io.gDecoderIn.valid := gDecoder.io.id_o.valid && gDecoder.io.id_o.bits.is_ball
  ballDomain.io.gDecoderIn.bits  := Mux(ballDomain.io.gDecoderIn.valid, gDecoder.io.id_o.bits, DontCare)
  
  // fence信号连接

// -----------------------------------------------------------------------------
// Backend: Mem Domain 包含DMA+TLB+SRAM的完整域
// -----------------------------------------------------------------------------
  val memDomain = Module(new MemDomain)
  
  // GlobalDecoder->MemDomain 
  memDomain.io.gDecoderIn.valid := gDecoder.io.id_o.valid && gDecoder.io.id_o.bits.is_mem
  memDomain.io.gDecoderIn.bits  := Mux(memDomain.io.gDecoderIn.valid, gDecoder.io.id_o.bits, DontCare)
  
  // 全局ready信号：只有对应的域ready时，gDecoder才ready
  gDecoder.io.id_o.ready := 
    (gDecoder.io.id_o.bits.is_ball && ballDomain.io.gDecoderIn.ready) ||
    (gDecoder.io.id_o.bits.is_mem && memDomain.io.gDecoderIn.ready)

// -----------------------------------------------------------------------------
// Backend: MemDomain Connections
// -----------------------------------------------------------------------------
  // MemDomain->DMA
  memDomain.io.dma.read.req <> outer.reader.module.io.req
  outer.reader.module.io.resp <> memDomain.io.dma.read.resp
  memDomain.io.dma.write.req <> outer.writer.module.io.req
  outer.writer.module.io.resp <> memDomain.io.dma.write.resp
  
  // DMA->TLB
  outer.reader.module.io.tlb <> tlb.io.clients(1)
  outer.writer.module.io.tlb <> tlb.io.clients(0)
  
  // 连接MemDomain的TLB接口 (暂时使用DontCare，后续可以连接)
  // memDomain.io.tlb := DontCare  

// -----------------------------------------------------------------------------
// Backend: Domain Bridge: BallDomain->MemDomain
// -----------------------------------------------------------------------------
  ballDomain.io.sramRead  <> memDomain.io.ballDomain.sramRead
  ballDomain.io.sramWrite <> memDomain.io.ballDomain.sramWrite
  ballDomain.io.accRead   <> memDomain.io.ballDomain.accRead
  ballDomain.io.accWrite  <> memDomain.io.ballDomain.accWrite

// ---------------------------------------------------------------------------
// 返回RoCC接口连接 - 合并BallDomain和MemDomain的响应
// ---------------------------------------------------------------------------
  // 优先级仲裁：BallDomain优先级高于MemDomain
  val respArb = Module(new Arbiter(new RoCCResponseBB()(p), 2))
  respArb.io.in(0) <> ballDomain.io.roccResp
  respArb.io.in(1) <> memDomain.io.roccResp
  io.resp <> respArb.io.out


// ---------------------------------------------------------------------------
// CSR 寄存器处理
// ---------------------------------------------------------------------------
  val fenceCSR = FenceCSR()
  val fenceSet = ballDomain.io.fence_o
  val allDomainsIdle = !ballDomain.io.busy && !memDomain.io.busy
  when (fenceSet) {
    fenceCSR := 1.U
    io.cmd.ready :=allDomainsIdle
  }
  when (fenceCSR =/= 0.U) {
    when(allDomainsIdle) {
      fenceCSR := 0.U
    }.otherwise{
      io.cmd.ready := false.B
    }
  }
  io.busy := fenceCSR =/= 0.U
}
