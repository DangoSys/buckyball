package framework.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tile._
import framework.balldomain.blink.{BankRead, BankWrite}
import freechips.rocketchip.tilelink.TLEdgeOut
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}

import framework.memdomain.frontend.MemController
import framework.memdomain.frontend.outside_channel.dma.{
  BBReadRequest,
  BBReadResponse,
  BBStreamReader,
  BBStreamReaderParam,
  BBStreamWriter,
  BBStreamWriterParam,
  BBWriteRequest,
  BBWriteResponse
}
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBExceptionIO, BBTLBIO, BBTLBPTWIO}

import framework.memdomain.midend.MemScheduler
import framework.memdomain.backend.MemManager
import framework.top.GlobalConfig

@instantiable
class MemDomain(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {

  @public
  val io = IO(new Bundle {
    // -------------------------------------------------
    // Command Channel
    // -------------------------------------------------
    // global RS -> MemDomain
    val global_issue_i    = Flipped(Decoupled(new GlobalRsIssue(b)))
    // MemDomain -> global RS
    val global_complete_o = Decoupled(new GlobalRsComplete(b))
    val busy              = Output(Bool())

    // -------------------------------------------------
    // Inside Channel
    // -------------------------------------------------
    // Bank interface for interaction with Ball Domain
    val ballDomain = new Bundle {
      val bankRead  = Vec(b.memDomain.bankNum, new BankRead(b))
      val bankWrite = Vec(b.memDomain.bankNum, new BankWrite(b))

    }

    // -------------------------------------------------
    // Outside Channel
    // -------------------------------------------------
    val dma = new Bundle {

      val read = new Bundle {
        val req  = Decoupled(new BBReadRequest())
        val resp = Flipped(Decoupled(new BBReadResponse(b.memDomain.bankWidth)))
      }

      val write = new Bundle {
        val req  = Decoupled(new BBWriteRequest(b.memDomain.bankWidth))
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }

    }

    val tlb    = Vec(2, Flipped(new BBTLBIO(b)))
    val ptw    = Vec(1, new BBTLBPTWIO(b))
    val tlbExp = Vec(1, new BBTLBExceptionIO)
  })

  val frontend: Instance[MemController] = Instantiate(new MemController(b)(edge))
  val midend:   Instance[MemScheduler]  = Instantiate(new MemScheduler(b))
  val backend:  Instance[MemManager]    = Instantiate(new MemManager(b))

  // -------------------------------------------------
  // Connection with outside (all in frontend)
  // -------------------------------------------------
  // Global RS interface
  frontend.io.global_issue_i <> io.global_issue_i
  frontend.io.global_complete_o <> io.global_complete_o
  frontend.io.busy := io.busy

  // DMA interface
  frontend.io.intradma <> io.dma

  // TLB interface
  frontend.io.tlb <> io.tlb
  frontend.io.ptw <> io.ptw
  frontend.io.tlbExp <> io.tlbExp

  // Ball Domain interface (connects to frontend's interdma)
  frontend.io.interdma.bankRead <> io.ballDomain.bankRead
  frontend.io.interdma.bankWrite <> io.ballDomain.bankWrite

  // -------------------------------------------------
  // Internal Connection (frontend - midend - backend)
  // -------------------------------------------------
  // Frontend to Midend: route interdma requests to scheduler
  midend.io.frontend.bankRead <> frontend.io.interdma.bankRead
  midend.io.frontend.bankWrite <> frontend.io.interdma.bankWrite

  // Midend to Backend: route scheduled requests to memory manager
  midend.io.mem_req <> backend.io.mem_req

  // -------------------------------------------------
  // DMA Modules (Reader and Writer) - Internal instantiation
  // -------------------------------------------------
  val readerParam = BBStreamReaderParam(
    nXacts = b.memDomain.dma_n_xacts,
    beatBits = b.memDomain.dma_buswidth,
    maxBytes = b.memDomain.dma_maxbytes,
    dataWidth = b.memDomain.dma_buswidth,
    tledge = edge
  )

  val writerParam = BBStreamWriterParam(
    nXacts = b.memDomain.dma_n_xacts,
    beatBits = b.memDomain.dma_buswidth,
    maxBytes = b.memDomain.dma_maxbytes,
    dataWidth = b.memDomain.dma_buswidth,
    tledge = edge
  )

  val reader: Instance[BBStreamReader] = Instantiate(new BBStreamReader(readerParam, b))
  val writer: Instance[BBStreamWriter] = Instantiate(new BBStreamWriter(writerParam, b))

  // Connect DMA to TLB and interfaces
  reader.io.tlb <> io.tlb(1)
  writer.io.tlb <> io.tlb(0)

  // Connect DMA to external DMA interface
  io.dma.read.req <> reader.io.req
  reader.io.resp <> io.dma.read.resp
  io.dma.write.req <> writer.io.req
  writer.io.resp <> io.dma.write.resp

  // Connect DMA flush signals to TLB exceptions
  reader.io.flush := io.tlbExp(0).flush()
  writer.io.flush := io.tlbExp(0).flush()
}
