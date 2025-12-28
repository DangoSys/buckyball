package framework.memdomain.frontend

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import framework.memdomain.frontend.outside_channel.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.memdomain.frontend.outside_channel.{MemLoader, MemStorer}
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBCluster, BBTLBExceptionIO, BBTLBIO, BBTLBPTWIO}
import freechips.rocketchip.tilelink.TLEdgeOut
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.decoder.MemDomainDecoder
import framework.memdomain.frontend.cmd_channel.rs.MemReservationStation
import framework.memdomain.utils.pmc.MemCyclePMC

/**
 * MemController: Controller that encapsulates scratchpad and accumulator
 * Provides DMA interface and Ball Domain interface
 */
@instantiable
class MemController(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {

  @public
  val io = IO(new Bundle {
    // Issue interface from global RS (single channel)
    val global_issue_i    = Flipped(Decoupled(new GlobalRsIssue(b)))
    // Report completion to global RS (single channel)
    val global_complete_o = Decoupled(new GlobalRsComplete(b))

    // Bank read/write interface - used by load/store
    val interdma = new Bundle {
      val bankRead  = Vec(b.memDomain.bankNum, Flipped(new BankRead(b)))
      val bankWrite = Vec(b.memDomain.bankNum, Flipped(new BankWrite(b)))
    }

    // DMA interface - used by Outer DRAM controller
    val intradma = new Bundle {

      val read = new Bundle {
        val req  = Decoupled(new BBReadRequest())
        val resp = Flipped(Decoupled(new BBReadResponse(b.memDomain.bankWidth)))
      }

      val write = new Bundle {
        val req  = Decoupled(new BBWriteRequest(b.memDomain.bankWidth))
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }

    }

    // TLB interface - exposed externally for DMA modules (BBStreamReader/BBStreamWriter)
    // TLB connection is managed internally, but interface needs to be exposed for external DMA
    val tlb    = Vec(2, Flipped(new BBTLBIO(b)))
    // PTW interface - needs to connect to upper level PTW (shared TLB has only 1 PTW)
    val ptw    = Vec(1, new BBTLBPTWIO(b))
    // TLB exception interface - exposed to upper level for handling flush, etc. (shared TLB has only 1 exp)
    val tlbExp = Vec(1, new BBTLBExceptionIO)
    // Busy signal
    val busy   = Output(Bool())
  })

  val memDecoder: Instance[MemDomainDecoder]      = Instantiate(new MemDomainDecoder(b))
  val memRs:      Instance[MemReservationStation] = Instantiate(new MemReservationStation(b))
  val memLoader:  Instance[MemLoader]             = Instantiate(new MemLoader(b))
  val memStorer:  Instance[MemStorer]             = Instantiate(new MemStorer(b))
  val pmc:        Instance[MemCyclePMC]           = Instantiate(new MemCyclePMC(b))
  // TLB cluster - internal TLB management for DMA modules
  // Supports 2 clients: BBStreamReader (client 1) and BBStreamWriter (client 0)
  val tlbCluster =
    Instantiate(new BBTLBCluster(b)(edge))

// -----------------------------------------------------------------------------
// Global RS -> MemDecoder
// -----------------------------------------------------------------------------
  memDecoder.io.raw_cmd_i.valid := io.global_issue_i.valid
  memDecoder.io.raw_cmd_i.bits  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready       := memDecoder.io.raw_cmd_i.ready

// -----------------------------------------------------------------------------
// MemDecoder -> MemReservationStation
// -----------------------------------------------------------------------------
  // Connect decoded instruction and global rob_id
  memRs.io.mem_decode_cmd_i.valid       := memDecoder.io.mem_decode_cmd_o.valid
  memRs.io.mem_decode_cmd_i.bits.cmd    := memDecoder.io.mem_decode_cmd_o.bits
  memRs.io.mem_decode_cmd_i.bits.rob_id := io.global_issue_i.bits.rob_id
  memDecoder.io.mem_decode_cmd_o.ready  := memRs.io.mem_decode_cmd_i.ready

// -----------------------------------------------------------------------------
// MemReservationStation -> MemLoader/MemStorer
// -----------------------------------------------------------------------------
  memLoader.io.cmdReq <> memRs.io.issue_o.ld
  memStorer.io.cmdReq <> memRs.io.issue_o.st
  memRs.io.commit_i.ld <> memLoader.io.cmdResp
  memRs.io.commit_i.st <> memStorer.io.cmdResp

//-----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  pmc.io.ldReq_i.valid  := memRs.io.issue_o.ld.fire
  pmc.io.ldReq_i.bits   := memRs.io.issue_o.ld.bits
  pmc.io.stReq_i.valid  := memRs.io.issue_o.st.fire
  pmc.io.stReq_i.bits   := memRs.io.issue_o.st.bits
  pmc.io.ldResp_o.valid := memLoader.io.cmdResp.fire
  pmc.io.ldResp_o.bits  := memLoader.io.cmdResp.bits
  pmc.io.stResp_o.valid := memStorer.io.cmdResp.fire
  pmc.io.stResp_o.bits  := memStorer.io.cmdResp.bits

  // Connect MemLoader and MemStorer to DMA interface
  memLoader.io.dmaReq <> io.intradma.read.req
  io.intradma.read.resp <> memLoader.io.dmaResp
  memStorer.io.dmaReq <> io.intradma.write.req
  io.intradma.write.resp <> memStorer.io.dmaResp

  // TLB connection - internal TLB cluster connected to external DMA modules
  // Client 0: BBStreamWriter, Client 1: BBStreamReader
  io.tlb <> tlbCluster.io.clients

  // PTW interface - connect to upper level page table walker
  io.ptw <> tlbCluster.io.ptw

  // TLB exception interface - connect to upper level for flush handling
  tlbCluster.io.exp <> io.tlbExp

  // Connect MemLoader and MemStorer to MemController's DMA interface
  memLoader.io.bankWrite <> io.interdma.bankWrite
  memStorer.io.bankRead <> io.interdma.bankRead

  // Completion signal connected to global RS
  io.global_complete_o <> memRs.io.complete_o

  // Busy signal
  // Simple busy signal
  io.busy := !memRs.io.complete_o.ready
}
