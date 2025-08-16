package examples.toy

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
import framework.builtin.mem.{Scratchpad, SimpleStreamReader, SimpleStreamWriter}
import framework.builtin.frontend.{FrontendTLB, GlobalDecoder}
// import framework.builtin.frontend.rs.ReservationStation
import framework.ballcore.ballcore._
// import framework.ballcore.ballcore.LazyRoCCBB
import framework.builtin.load.MemLoader
import framework.builtin.store.MemStorer
import examples.toy.balldomain.ExecuteController 
import examples.BuckyBallConfigs.CustomBuckyBallConfig 


class ToyBuckyBall(val bbconfig: CustomBuckyBallConfig)(implicit p: Parameters)
  extends LazyRoCC (opcodes = bbconfig.opcodes, nPTWPorts = 2) {

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

  override lazy val module = new ToyBuckyBallModule(this)

  // The LazyRoCC class contains two TLOutputNode instances, atlNode and tlNode. 
  // atlNode connects into a tile-local arbiter along with the backside of the L1 instruction cache. 
  // tlNode connects directly to the L1-L2 crossbar. 
  // The corresponding Tilelink ports in the module implementation's IO bundle are atl and tl, respectively.
  override val tlNode = id_node 
  override val atlNode = TLIdentityNode() 
  val node = tlNode 
}

class ToyBuckyBallModule(outer: ToyBuckyBall) extends LazyRoCCModuleImp(outer) 
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
// Frontend: Global Decode and Command Processing
// -----------------------------------------------------------------------------
  implicit val bbconfig: CustomBuckyBallConfig = outer.bbconfig
  val globalDecoder = Module(new GlobalDecoder)
  globalDecoder.io.id_i.valid    := io.cmd.valid
  globalDecoder.io.id_i.bits.cmd := io.cmd.bits
  io.cmd.ready                   := globalDecoder.io.id_i.ready

// -----------------------------------------------------------------------------
// Frontend: Domain Decoders
// -----------------------------------------------------------------------------
  val exDecoder = Module(new examples.toy.balldomain.ExDomainDecoder)
  val memDecoder = Module(new examples.toy.memdomain.MemDomainDecoder)
  
  // 连接GlobalDecoder到ExDecoder
  exDecoder.io.post_decode_cmd_i.valid := globalDecoder.io.id_rs.valid && globalDecoder.io.id_rs.bits.is_ex
  exDecoder.io.post_decode_cmd_i.bits := globalDecoder.io.id_rs.bits
  
  // 连接GlobalDecoder到MemDecoder  
  memDecoder.io.post_decode_cmd_i.valid := globalDecoder.io.id_rs.valid && globalDecoder.io.id_rs.bits.is_mem
  memDecoder.io.post_decode_cmd_i.bits := globalDecoder.io.id_rs.bits
  
  // 全局ready信号：只有相应的domain decoder ready，globalDecoder才ready
  globalDecoder.io.id_rs.ready := Mux(globalDecoder.io.id_rs.bits.is_ex, 
    exDecoder.io.post_decode_cmd_i.ready,
    memDecoder.io.post_decode_cmd_i.ready)

// -----------------------------------------------------------------------------
// Frontend: Domain-specific Reservation Stations
// -----------------------------------------------------------------------------
  // EX域专用保留栈
  val exReservationStation = Module(new examples.toy.balldomain.ExReservationStation)
  exReservationStation.io.ex_decode_cmd_i <> exDecoder.io.ex_decode_cmd_o
  
  // Mem域专用保留栈
  val memReservationStation = Module(new examples.toy.memdomain.MemReservationStation)
  memReservationStation.io.mem_decode_cmd_i <> memDecoder.io.mem_decode_cmd_o

// -----------------------------------------------------------------------------
// Backend: Load Controller
// -----------------------------------------------------------------------------
  val memLoader = Module(new MemLoader)
  memLoader.io.cmdReq <> memReservationStation.io.issue_o.ld
  memReservationStation.io.commit_i.ld <> memLoader.io.cmdResp
  
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
  memStorer.io.cmdReq <> memReservationStation.io.issue_o.st
  memReservationStation.io.commit_i.st <> memStorer.io.cmdResp
  
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
  exec.io.cmdReq <> exReservationStation.io.issue_o
  exReservationStation.io.commit_i <> exec.io.cmdResp
  
  // 连接ExecuteController到Scratchpad的专用执行接口
  exec.io.sramRead <> spad.io.exec.sramread
  exec.io.sramWrite <> spad.io.exec.sramwrite
  exec.io.accRead <> spad.io.exec.accread
  exec.io.accWrite <> spad.io.exec.accwrite

//---------------------------------------------------------------------------
// 返回RoCC接口连接 - 合并两个保留栈的响应
//---------------------------------------------------------------------------
  // 优先级仲裁：EX域优先级高于Mem域
  val respArb = Module(new Arbiter(new RoCCResponse()(p), 2))
  respArb.io.in(0) <> exReservationStation.io.rs_rocc_o.resp
  respArb.io.in(1) <> memReservationStation.io.rs_rocc_o.resp
  io.resp <> respArb.io.out
  
  // 只要任一保留栈忙碌，整个系统就忙碌
  io.busy := exReservationStation.io.rs_rocc_o.busy || memReservationStation.io.rs_rocc_o.busy

}
