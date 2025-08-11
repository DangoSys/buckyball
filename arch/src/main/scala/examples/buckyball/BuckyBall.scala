package buckyball

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tile._
import freechips.rocketchip.util.ClockGate
import freechips.rocketchip.tilelink._
import BBISA._
import mem.{Scratchpad, SimpleStreamReader, SimpleStreamWriter}
import frontend.{FrontendTLB, Decoder}
import frontend.rs.ReservationStation
import framework.ballcore.ballcore._
import framework.ballcore.ballcore.LazyRoCCBB
import buckyball.load.MemLoader
import buckyball.store.MemStorer
import buckyball.exec.ExecuteController


class BuckyBall(val bbconfig: BuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCCBB (opcodes = bbconfig.opcodes, nPTWPorts = 2) {

  val xLen = p(TileKey).core.xLen   // the width of core's register file
  
  // DMA组件现在在BuckyBall层面
  val id_node = TLIdentityNode()
  val xbar_node = TLXbar()
  
  val spad_w = bbconfig.inputType.getWidth * bbconfig.veclane
  val reader = LazyModule(new SimpleStreamReader(bbconfig.max_in_flight_mem_reqs, bbconfig.dma_buswidth, bbconfig.dma_maxbytes, spad_w))
  val writer = LazyModule(new SimpleStreamWriter(bbconfig.max_in_flight_mem_reqs, bbconfig.dma_buswidth, bbconfig.dma_maxbytes, spad_w))

  xbar_node := TLBuffer() := reader.node
  xbar_node := TLBuffer() := writer.node
  id_node := TLWidthWidget(bbconfig.dma_buswidth/8) := TLBuffer() := xbar_node

  override lazy val module = new BuckyBallModule(this)

  // The LazyRoCC class contains two TLOutputNode instances, atlNode and tlNode. 
  // atlNode connects into a tile-local arbiter along with the backside of the L1 instruction cache. 
  // tlNode connects directly to the L1-L2 crossbar. 
  // The corresponding Tilelink ports in the module implementation's IO bundle are atl and tl, respectively.
  override val tlNode = id_node 
  override val atlNode = TLIdentityNode() 
  val node = tlNode 
}

class BuckyBallModule(outer: BuckyBall) extends LazyRoCCModuleImpBB(outer) 
  with HasCoreParameters {
  import outer.bbconfig._
  
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
// Memory: Scratchpad (纯粹的SRAM banks)
// -----------------------------------------------------------------------------
  val spad = Module(new Scratchpad(outer.bbconfig))

// -----------------------------------------------------------------------------
// Frontend: Decode and Command Processing
// -----------------------------------------------------------------------------
  implicit val bbconfig: BuckyBallConfig = outer.bbconfig
  val decoder = Module(new Decoder)
  decoder.io.id_i.valid    := io.cmd.valid
  decoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready             := decoder.io.id_i.ready

// -----------------------------------------------------------------------------
// Frontend: Reservation Station with integrated RoB
// -----------------------------------------------------------------------------
  val rs = Module(new ReservationStation)
  decoder.io.id_rs <> rs.io.id_i

// -----------------------------------------------------------------------------
// Backend: Load Controller
// -----------------------------------------------------------------------------
  val memLoader = Module(new MemLoader)
  memLoader.io.cmdReq <> rs.io.issue_o.ld
  rs.io.commit_i.ld <> memLoader.io.cmdResp
  
  // 连接MemLoader直接到SimpleStreamReader
  memLoader.io.dmaReq <> outer.reader.module.io.req
  outer.reader.module.io.resp <> memLoader.io.dmaResp
  
  // 连接DMA Reader的TLB到TLB (client 1) - DMA内部做地址翻译
  outer.reader.module.io.tlb <> tlb.io.clients(1)
  
  // 连接MemLoader到Scratchpad SRAM写入接口
  memLoader.io.sramWrite <> spad.io.dma.sramwrite
  memLoader.io.accWrite <> spad.io.dma.accwrite

// -----------------------------------------------------------------------------
// Backend: Store Controller
// -----------------------------------------------------------------------------
  val memStorer = Module(new MemStorer)
  memStorer.io.cmdReq <> rs.io.issue_o.st
  rs.io.commit_i.st <> memStorer.io.cmdResp
  
  // 连接MemStorer直接到SimpleStreamWriter
  memStorer.io.dmaReq <> outer.writer.module.io.req
  outer.writer.module.io.resp <> memStorer.io.dmaResp
  
  // 连接DMA Writer的TLB到TLB (client 0) - DMA内部做地址翻译
  outer.writer.module.io.tlb <> tlb.io.clients(0)
  
  // 连接MemStorer到Scratchpad SRAM读取接口
  memStorer.io.sramRead <> spad.io.dma.sramread
  memStorer.io.accRead  <> spad.io.dma.accread

// -----------------------------------------------------------------------------
// Backend: Execute Controller
// -----------------------------------------------------------------------------
  val exec = Module(new ExecuteController)
  exec.io.cmdReq <> rs.io.issue_o.ex
  rs.io.commit_i.ex <> exec.io.cmdResp
  
  // 连接ExecuteController到Scratchpad的专用执行接口
  exec.io.sramRead <> spad.io.exec.sramread
  exec.io.sramWrite <> spad.io.exec.sramwrite
  exec.io.accRead <> spad.io.exec.accread
  exec.io.accWrite <> spad.io.exec.accwrite

//---------------------------------------------------------------------------
// 返回RoCC接口连接
//---------------------------------------------------------------------------
  io.resp <> rs.io.rs_rocc_o.resp
  io.busy := rs.io.rs_rocc_o.busy

}
