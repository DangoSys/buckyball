package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.PostGDCmd
import freechips.rocketchip.tile._
import framework.builtin.frontend.FrontendTLBIO
import framework.builtin.memdomain.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.memdomain.{MemLoader, MemStorer, MemController}
import framework.rocket.RoCCResponseBB

class MemDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // 来自GlobalDecoder的输入
    val gDecoderIn = Flipped(Decoupled(new PostGDCmd))
    
    // 与Ball Domain交互的SRAM接口
    val ballDomain = new Bundle {
      val sramRead = Vec(b.sp_banks, new SramReadIO(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(b.sp_banks, new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
      val accRead = Vec(b.acc_banks, new SramReadIO(b.acc_bank_entries, b.acc_w))
      val accWrite = Vec(b.acc_banks, new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len))
    }
    
    // DMA接口
    val dma = new Bundle {
      val read = new Bundle {
        val req = Decoupled(new BBReadRequest())
        val resp = Flipped(Decoupled(new BBReadResponse(b.spad_w)))
      }
      val write = new Bundle {
        val req = Decoupled(new BBWriteRequest(b.spad_w))
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }
    }
    
    // TLB接口
    // val tlb = new FrontendTLBIO()(p)
    
    // RoCC响应接口
    val roccResp = Decoupled(new RoCCResponseBB()(p))
    val busy = Output(Bool())
  })

  // 内部组件实例化
  val memDecoder = Module(new MemDomainDecoder)
  val memRs = Module(new MemReservationStation)
  val memLoader = Module(new MemLoader)
  val memStorer = Module(new MemStorer)
  
  // 内部MemController (封装了spad和acc)
  val memController = Module(new MemController)

  // 连接GlobalDecoder到MemDecoder
  memDecoder.io.raw_cmd_i <> io.gDecoderIn
  
  // 连接MemDecoder到ReservationStation
  memRs.io.mem_decode_cmd_i <> memDecoder.io.mem_decode_cmd_o
  
  // 连接ReservationStation到MemLoader和MemStorer
  memLoader.io.cmdReq <> memRs.io.issue_o.ld
  memStorer.io.cmdReq <> memRs.io.issue_o.st
  memRs.io.commit_i.ld <> memLoader.io.cmdResp
  memRs.io.commit_i.st <> memStorer.io.cmdResp
  
  // 连接MemLoader和MemStorer到DMA
  memLoader.io.dmaReq <> io.dma.read.req
  io.dma.read.resp <> memLoader.io.dmaResp
  memStorer.io.dmaReq <> io.dma.write.req
  io.dma.write.resp <> memStorer.io.dmaResp
  
  // 连接TLB (这里需要根据实际的TLB接口连接，暂时注释)
  // memLoader.io.tlb <> io.tlb
  // memStorer.io.tlb <> io.tlb
  
  // 连接MemLoader和MemStorer到MemController的DMA接口
  memLoader.io.sramWrite <> memController.io.dma.sramWrite
  memLoader.io.accWrite <> memController.io.dma.accWrite
  memStorer.io.sramRead <> memController.io.dma.sramRead
  memStorer.io.accRead <> memController.io.dma.accRead
  
  // Ball Domain SRAM接口连接到MemController的Ball Domain接口
  io.ballDomain.sramRead <> memController.io.ballDomain.sramRead
  io.ballDomain.sramWrite <> memController.io.ballDomain.sramWrite
  io.ballDomain.accRead <> memController.io.ballDomain.accRead
  io.ballDomain.accWrite <> memController.io.ballDomain.accWrite
  
  // RoCC响应直接连接
  io.roccResp <> memRs.io.rs_rocc_o.resp
  
  // 忙碌信号
  io.busy := memRs.io.rs_rocc_o.busy
}