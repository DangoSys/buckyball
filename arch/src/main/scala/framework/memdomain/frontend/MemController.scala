package framework.memdomain.frontend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.decoder.PostGDCmd
import freechips.rocketchip.tile._
import framework.memdomain.frontend.outside_channel.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.memdomain.frontend.outside_channel.{MemLoader, MemStorer}
import framework.memdomain.frontend.cmd_channel.rs.MemReservationStation
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBCluster, BBTLBExceptionIO, BBTLBIO}
import framework.memdomain.utils.pmc.MemCyclePMC
import freechips.rocketchip.tilelink.TLEdgeOut
import freechips.rocketchip.rocket.TLBPTWIO
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import framework.memdomain.MemDomainParam

/**
 * MemController: Controller that encapsulates scratchpad and accumulator
 * Provides DMA interface and Ball Domain interface
 */
@instantiable
class MemController(val parameter: MemDomainParam)(edge: TLEdgeOut)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {
    // Issue interface from global RS (single channel)
    val global_issue_i = Flipped(Decoupled(new GlobalRsIssue(parameter.rob_entries)))

    // Report completion to global RS (single channel)
    val global_complete_o = Decoupled(new GlobalRsComplete(parameter.rob_entries))

    // Bank read/write interface - used by load/store
    val interdma = new Bundle {

      val bankRead = Vec(
        parameter.bankNum,
        Flipped(new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.bankNum))
      )

      val bankWrite = Vec(
        parameter.bankNum,
        Flipped(new BankWrite(
          parameter.bankEntries,
          parameter.bankWidth,
          parameter.bankMaskLen,
          parameter.rob_entries,
          parameter.bankNum
        ))
      )

    }

    // DMA interface - used by Outer DRAM controller
    val intradma = new Bundle {

      val read = new Bundle {
        val req  = Decoupled(new BBReadRequest())
        val resp = Flipped(Decoupled(new BBReadResponse(parameter.bankWidth)))
      }

      val write = new Bundle {
        val req  = Decoupled(new BBWriteRequest(parameter.bankWidth))
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }

    }

    // TLB interface - exposed externally for DMA modules (BBStreamReader/BBStreamWriter)
    // TLB connection is managed internally, but interface needs to be exposed for external DMA
    val tlb = Vec(2, Flipped(new BBTLBIO))

    // PTW interface - needs to connect to upper level PTW (shared TLB has only 1 PTW)
    val ptw = Vec(1, new TLBPTWIO)

    // TLB exception interface - exposed to upper level for handling flush, etc. (shared TLB has only 1 exp)
    val tlbExp = Vec(1, new BBTLBExceptionIO)

    // Busy signal
    val busy = Output(Bool())
  })

  val memDecoder: Instance[framework.memdomain.frontend.cmd_channel.decoder.MemDomainDecoder] =
    Instantiate(new framework.memdomain.frontend.cmd_channel.decoder.MemDomainDecoder(parameter))
  val memRs:      Instance[framework.memdomain.frontend.cmd_channel.rs.MemReservationStation] =
    Instantiate(new framework.memdomain.frontend.cmd_channel.rs.MemReservationStation(parameter))
  val memLoader:  Instance[framework.memdomain.frontend.outside_channel.MemLoader]            =
    Instantiate(new framework.memdomain.frontend.outside_channel.MemLoader(parameter))
  val memStorer:  Instance[framework.memdomain.frontend.outside_channel.MemStorer]            =
    Instantiate(new framework.memdomain.frontend.outside_channel.MemStorer(parameter))

  // TLB cluster - internal TLB management for DMA modules
  // Supports 2 clients: BBStreamReader (client 1) and BBStreamWriter (client 0)
  val tlbCluster =
    Module(new BBTLBCluster(2, parameter.tlb_size, parameter.dma_maxbytes, use_shared_tlb = true)(edge, p))

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
  val pmc: Instance[framework.memdomain.utils.pmc.MemCyclePMC] =
    Instantiate(new framework.memdomain.utils.pmc.MemCyclePMC(parameter))
  pmc.io.ldReq_i.valid  := memRs.io.issue_o.ld.fire
  pmc.io.ldReq_i.bits   := memRs.io.issue_o.ld.bits
  pmc.io.stReq_i.valid  := memRs.io.issue_o.st.fire
  pmc.io.stReq_i.bits   := memRs.io.issue_o.st.bits
  pmc.io.ldResp_o.valid := memLoader.io.cmdResp.fire
  pmc.io.ldResp_o.bits  := memLoader.io.cmdResp.bits
  pmc.io.stResp_o.valid := memStorer.io.cmdResp.fire
  pmc.io.stResp_o.bits  := memStorer.io.cmdResp.bits
//
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
