package framework.memdomain.frontend

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile._
import framework.memdomain.frontend.outside_channel.dma.{StreamReader, StreamWriter}
import framework.memdomain.frontend.outside_channel.{MemConfiger, MemConfigerIO, MemLoader, MemStorer}
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBCluster, BBTLBExceptionIO, BBTLBIO, BBTLBPTWIO}
import freechips.rocketchip.tilelink.{TLBundle, TLEdgeOut}
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.decoder.MemDomainDecoder
import framework.memdomain.frontend.cmd_channel.rs.MemReservationStation
import framework.memdomain.utils.pmc.MemCyclePMC

/**
 * MemFrontend:
 * Provides DMA interface and Ball Domain interface
 */
@instantiable
class MemFrontend(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {

  @public
  val io = IO(new Bundle {
    // Issue interface from global RS (single channel)
    val global_issue_i    = Flipped(Decoupled(new GlobalRsIssue(b)))
    // Report completion to global RS (single channel)
    val global_complete_o = Decoupled(new GlobalRsComplete(b))

    // Bank read/write interface - used by load/store
    val interdma = new Bundle {
      val bankRead  = Flipped(new BankRead(b))
      val bankWrite = Flipped(new BankWrite(b))
    }

    // TLB interfaces for internal DMA modules (Reader/Writer)
    // These are NOT exposed to outside - only PTW and TLB exception are exposed
    // PTW interface - needs to connect to upper level PTW (shared TLB has only 1 PTW)
    val ptw    = Vec(1, new BBTLBPTWIO(b))
    // TLB exception interface - exposed to upper level for handling flush, etc. (shared TLB has only 1 exp)
    val tlbExp = Vec(1, new BBTLBExceptionIO)

    // TileLink physical connections for DMA (Reader/Writer)
    val tl_reader = new TLBundle(edge.bundle)
    val tl_writer = new TLBundle(edge.bundle)

    val config = Decoupled(new MemConfigerIO(b))

    // Busy signal
    val busy = Output(Bool())
  })

  val memDecoder: Instance[MemDomainDecoder]      = Instantiate(new MemDomainDecoder(b))
  val memRs:      Instance[MemReservationStation] = Instantiate(new MemReservationStation(b))
  val memLoader:  Instance[MemLoader]             = Instantiate(new MemLoader(b))
  val memStorer:  Instance[MemStorer]             = Instantiate(new MemStorer(b))
  val pmc:        Instance[MemCyclePMC]           = Instantiate(new MemCyclePMC(b))

  // TLB cluster - internal TLB management for DMA modules
  // Supports 2 clients: StreamReader (client 1) and StreamWriter (client 0)
  val tlbCluster = Instantiate(new BBTLBCluster(b)(edge))

  // DMA Reader and Writer modules - handle actual DMA transfers
  val reader:   Instance[StreamReader] = Instantiate(new StreamReader(b)(edge))
  val writer:   Instance[StreamWriter] = Instantiate(new StreamWriter(b)(edge))
  val configer: Instance[MemConfiger]  = Instantiate(new MemConfiger(b))

// -----------------------------------------------------------------------------
// Global RS -> MemDecoder
// -----------------------------------------------------------------------------
  memDecoder.io.cmd_i.valid := io.global_issue_i.valid
  memDecoder.io.cmd_i.bits  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready   := memDecoder.io.cmd_i.ready
  io.config <> configer.io.config

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
  configer.io.cmdReq <> memRs.io.issue_o.cf
  memRs.io.commit_i.ld <> memLoader.io.cmdResp
  memRs.io.commit_i.st <> memStorer.io.cmdResp
  memRs.io.commit_i.cf <> configer.io.cmdResp

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

  // Connect Reader and Writer to MemLoader and MemStorer
  memLoader.io.dmaReq <> reader.io.req
  reader.io.resp <> memLoader.io.dmaResp
  memStorer.io.dmaReq <> writer.io.req
  writer.io.resp <> memStorer.io.dmaResp

  // TLB connection - internal TLB cluster connected to DMA modules
  // Client 0: StreamWriter, Client 1: StreamReader
  // Insert pipeline registers to break combinational loops
  tlbCluster.io.clients(1).req.valid := reader.io.tlb.req.valid
  tlbCluster.io.clients(1).req.bits  := reader.io.tlb.req.bits
  reader.io.tlb.req.ready            := tlbCluster.io.clients(1).req.ready

  reader.io.tlb.resp.valid            := tlbCluster.io.clients(1).resp.valid
  reader.io.tlb.resp.bits             := tlbCluster.io.clients(1).resp.bits
  tlbCluster.io.clients(1).resp.ready := reader.io.tlb.resp.ready

  tlbCluster.io.clients(0).req.valid := writer.io.tlb.req.valid
  tlbCluster.io.clients(0).req.bits  := writer.io.tlb.req.bits
  writer.io.tlb.req.ready            := tlbCluster.io.clients(0).req.ready

  writer.io.tlb.resp.valid            := tlbCluster.io.clients(0).resp.valid
  writer.io.tlb.resp.bits             := tlbCluster.io.clients(0).resp.bits
  tlbCluster.io.clients(0).resp.ready := writer.io.tlb.resp.ready

  // Connect DMA flush signals to TLB exceptions
  reader.io.flush := io.tlbExp(0).flush()
  writer.io.flush := io.tlbExp(0).flush()

  // PTW interface - connect to upper level page table walker
  io.ptw <> tlbCluster.io.ptw

  // TLB exception interface - connect to upper level for flush handling
  tlbCluster.io.exp <> io.tlbExp

  // Connect TileLink physical ports from Reader/Writer to external interface
  io.tl_reader <> reader.io.tl
  io.tl_writer <> writer.io.tl

  // Connect MemLoader and MemStorer to MemController's DMA interface
  memLoader.io.bankWrite <> io.interdma.bankWrite
  memStorer.io.bankRead <> io.interdma.bankRead

  // Completion signal connected to global RS
  io.global_complete_o <> memRs.io.complete_o

  // Busy signal
  // Simple busy signal
  io.busy := !memRs.io.complete_o.ready
}
