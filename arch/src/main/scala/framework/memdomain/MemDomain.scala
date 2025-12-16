package framework.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.decoder.PostGDCmd
import freechips.rocketchip.tile._
import framework.memdomain.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.memdomain.mem.{SramReadIO, SramWriteIO, Scratchpad}
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

  val memController = Module(new MemController)
  val translator = Module(new Translator)
  val spad = Module(new Scratchpad(b))

// -----------------------------------------------------------------------------
// Global RS -> MemDecoder
// -----------------------------------------------------------------------------
  memController.io.global_issue_i.valid := io.global_issue_i.valid
  memController.io.global_issue_i.bits.cmd  := io.global_issue_i.bits.cmd
  io.global_issue_i.ready := memController.io.global_issue_i.ready

// -----------------------------------------------------------------------------
// MemDecoder -> MemReservationStation
// -----------------------------------------------------------------------------
  // Connect decoded instruction and global rob_id
  memController.io.global_issue_i.bits.rob_id := io.global_issue_i.bits.rob_id


  // Connect MemLoader and MemStorer to DMA
  memController.io.intradma.read.req <> io.dma.read.req
  io.dma.read.resp <> memController.io.intradma.read.resp
  memController.io.intradma.write.req <> io.dma.write.req
  io.dma.write.resp <> memController.io.intradma.write.resp

  // Connect TLB - now using internal BBTLBCluster
  io.tlb <> memController.io.tlb
  io.ptw <> memController.io.ptw

  // Connect exception interface - note direction: internal TLB's exp is Output, external interface is Input
  memController.io.tlbExp <> io.tlbExp

  // Connect MemLoader and MemStorer to MemController's DMA interface

  memController.io.interdma.sramwrite <> translator.io.dma_in.sramwrite
  memController.io.interdma.accwrite <>translator.io.dma_in.accwrite
  memController.io.interdma.sramread <> translator.io.dma_in.sramread
  memController.io.interdma.accread <> translator.io.dma_in.accread

  translator.io.dma_out.sramwrite <> spad.io.dma.sramwrite
  translator.io.dma_out.accwrite <> spad.io.dma.accwrite
  translator.io.dma_out.sramread <> spad.io.dma.sramread
  translator.io.dma_out.accread <> spad.io.dma.accread

  // ToPhysical interface
  // Ball Domain SRAM interface connected to MemController's Ball Domain interface
  val toPhysicalLines = Module(new ToPhysicalLine()(b, p))

  toPhysicalLines.io.sramRead_i  <> io.ballDomain.sramRead
  toPhysicalLines.io.sramWrite_i <> io.ballDomain.sramWrite

  toPhysicalLines.io.sramRead_o <> translator.io.exec_in.sramread
  toPhysicalLines.io.sramWrite_o <> translator.io.exec_in.sramwrite
  toPhysicalLines.io.accRead_o <> translator.io.exec_in.accread
  toPhysicalLines.io.accWrite_o <> translator.io.exec_in.accwrite

  translator.io.exec_out.sramread <> spad.io.exec.sramread
  translator.io.exec_out.sramwrite <> spad.io.exec.sramwrite
  translator.io.exec_out.accread <> spad.io.exec.accread
  translator.io.exec_out.accwrite <> spad.io.exec.accwrite

  // translator.io.dma_return := DontCare 
  // translator.io.exec_return := DontCare
  
  // Completion signal connected to global RS
  io.global_complete_o <> memController.io.global_complete_o

  // Busy signal
  // Simple busy signal
  io.busy := memController.io.busy
}
