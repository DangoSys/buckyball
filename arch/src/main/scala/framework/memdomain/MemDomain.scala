package framework.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.decoder.PostGDCmd
import freechips.rocketchip.tile._
import framework.memdomain.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.memdomain.{MemLoader, MemStorer, MemController}
import framework.memdomain.rs.MemReservationStation
import framework.memdomain.tlb.{BBTLBCluster, BBTLBIO, BBTLBExceptionIO}
import framework.memdomain.pmc.MemCyclePMC
import freechips.rocketchip.tilelink.TLEdgeOut
import freechips.rocketchip.rocket.TLBPTWIO
import framework.frontend.globalrs.{GlobalRsIssue, GlobalRsComplete}
import framework.balldomain.blink.{SramReadWithInfo, SramWriteWithInfo}
import framework.switcher.{ToPhysicalLine, ToVirtualLine}

class MemDomain(implicit b: CustomBuckyballConfig, p: Parameters, edge: TLEdgeOut) extends Module {
  private val numBanks = b.sp_banks + b.acc_banks
  val io = IO(new Bundle {
    // Issue interface from global RS (single channel)
    val global_issue_i = Flipped(Decoupled(new GlobalRsIssue))

    // Report completion to global RS (single channel)
    val global_complete_o = Decoupled(new GlobalRsComplete)

    // SRAM interface for interaction with Ball Domain
    val ballDomain = new Bundle {
      val sramRead = Vec(numBanks, new SramReadWithInfo(b.spad_bank_entries, b.spad_w))
      val sramWrite = Vec(numBanks, new SramWriteWithInfo(b.spad_bank_entries, b.spad_w, b.spad_mask_len))
    }

    // DMA interface
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

    // TLB interface - exposed externally for DMA use
    val tlb = Vec(2, Flipped(new BBTLBIO))

    // PTW interface - needs to connect to upper level PTW (shared TLB has only 1 PTW)
    val ptw = Vec(1, new TLBPTWIO)

    // TLB exception interface - exposed to upper level for handling flush, etc. (shared TLB has only 1 exp)
    val tlbExp = Vec(1, new BBTLBExceptionIO)

    // Busy signal
    val busy = Output(Bool())
  })

  val memDecoder = Module(new MemDomainDecoder)
  val memRs      = Module(new MemReservationStation)
  val memLoader  = Module(new MemLoader)
  val memStorer  = Module(new MemStorer)

  // Internal MemController (encapsulates spad and acc)
  val memController = Module(new MemController)

  // TLB cluster - use shared TLB like Gemmini
  val tlbCluster = Module(new BBTLBCluster(2, b.tlb_size, b.dma_maxbytes, use_shared_tlb = true))

// -----------------------------------------------------------------------------
// Global RS -> MemDecoder
// -----------------------------------------------------------------------------
  memDecoder.io.raw_cmd_i.valid := io.global_issue_i.valid
  memDecoder.io.raw_cmd_i.bits  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready := memDecoder.io.raw_cmd_i.ready

// -----------------------------------------------------------------------------
// MemDecoder -> MemReservationStation
// -----------------------------------------------------------------------------
  // Connect decoded instruction and global rob_id
  memRs.io.mem_decode_cmd_i.valid := memDecoder.io.mem_decode_cmd_o.valid
  memRs.io.mem_decode_cmd_i.bits.cmd := memDecoder.io.mem_decode_cmd_o.bits
  memRs.io.mem_decode_cmd_i.bits.rob_id := io.global_issue_i.bits.rob_id
  memDecoder.io.mem_decode_cmd_o.ready := memRs.io.mem_decode_cmd_i.ready

// -----------------------------------------------------------------------------
// MemReservationStation -> MemLoader/MemStorer
// -----------------------------------------------------------------------------
  memLoader.io.cmdReq <> memRs.io.issue_o.ld
  memStorer.io.cmdReq <> memRs.io.issue_o.st
  memRs.io.commit_i.ld <> memLoader.io.cmdResp
  memRs.io.commit_i.st <> memStorer.io.cmdResp

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  val pmc = Module(new MemCyclePMC)
  pmc.io.ldReq_i.valid := memRs.io.issue_o.ld.fire
  pmc.io.ldReq_i.bits := memRs.io.issue_o.ld.bits
  pmc.io.stReq_i.valid := memRs.io.issue_o.st.fire
  pmc.io.stReq_i.bits := memRs.io.issue_o.st.bits
  pmc.io.ldResp_o.valid := memLoader.io.cmdResp.fire
  pmc.io.ldResp_o.bits := memLoader.io.cmdResp.bits
  pmc.io.stResp_o.valid := memStorer.io.cmdResp.fire
  pmc.io.stResp_o.bits := memStorer.io.cmdResp.bits

  // Connect MemLoader and MemStorer to DMA
  memLoader.io.dmaReq <> io.dma.read.req
  io.dma.read.resp <> memLoader.io.dmaResp
  memStorer.io.dmaReq <> io.dma.write.req
  io.dma.write.resp <> memStorer.io.dmaResp

  // Connect TLB - now using internal BBTLBCluster
  io.tlb <> tlbCluster.io.clients
  io.ptw <> tlbCluster.io.ptw

  // Connect exception interface - note direction: internal TLB's exp is Output, external interface is Input
  tlbCluster.io.exp <> io.tlbExp

  // Connect MemLoader and MemStorer to MemController's DMA interface
  memLoader.io.sramWrite <> memController.io.dma.sramWrite
  memLoader.io.accWrite <> memController.io.dma.accWrite
  memStorer.io.sramRead <> memController.io.dma.sramRead
  memStorer.io.accRead <> memController.io.dma.accRead


  // ToPhysical interface
  // Ball Domain SRAM interface connected to MemController's Ball Domain interface
  val toPhysicalLines = Module(new ToPhysicalLine()(b, p))

  toPhysicalLines.io.sramRead_i  <> io.ballDomain.sramRead
  toPhysicalLines.io.sramWrite_i <> io.ballDomain.sramWrite

  toPhysicalLines.io.sramRead_o <> memController.io.ballDomain.sramRead
  toPhysicalLines.io.sramWrite_o <> memController.io.ballDomain.sramWrite
  toPhysicalLines.io.accRead_o <> memController.io.ballDomain.accRead
  toPhysicalLines.io.accWrite_o <> memController.io.ballDomain.accWrite

  // Completion signal connected to global RS
  io.global_complete_o <> memRs.io.complete_o

  // Busy signal
  // Simple busy signal
  io.busy := !memRs.io.complete_o.ready
}
