package framework.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.decoder.PostGDCmd
import freechips.rocketchip.tile._

import framework.balldomain.blink.{BankRead, BankWrite}
import freechips.rocketchip.tilelink.TLEdgeOut
import freechips.rocketchip.rocket.TLBPTWIO
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}

import framework.memdomain.frontend.MemController
import framework.memdomain.frontend.outside_channel.dma.{BBReadRequest, BBReadResponse, BBWriteRequest, BBWriteResponse}
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBExceptionIO, BBTLBIO}

import framework.memdomain.midend.MemScheduler
import framework.memdomain.backend.MemManager

object MemDomainParam {
  implicit def rw: upickle.default.ReadWriter[MemDomainParam] = upickle.default.macroRW

  /**
   * Load from JSON file
   */
  def fromJson(path: String): MemDomainParam = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    upickle.default.read[MemDomainParam](jsonStr)
  }

  /**
   * Generate from global config
   */
  def fromGlobal(global: framework.builtin.BaseConfig): MemDomainParam = {
    MemDomainParam(
      bankNum = global.bankNum,
      bankWidth = global.bankWidth,
      bankEntries = global.bankEntries,
      bankMaskLen = global.bankMaskLen,
      tlb_size = global.tlb_size,
      rob_entries = global.rob_entries,
      dma_maxbytes = global.dma_maxbytes,
      memAddrLen = global.memAddrLen,
      bankChannel = 8 // Default value
    )
  }

}

case class MemDomainParam(
  bankNum:      Int,
  bankWidth:    Int,
  bankEntries:  Int,
  bankMaskLen:  Int,
  tlb_size:     Int,
  rob_entries:  Int,
  dma_maxbytes: Int,
  memAddrLen:   Int,
  bankChannel:  Int)
    extends SerializableModuleParameter {
  override def toString: String =
    s"""MemDomainParam
       |  Bank num: $bankNum
       |  Bank width: $bankWidth bits
       |  Bank entries: $bankEntries
       |  Bank mask length: $bankMaskLen
       |  TLB size: $tlb_size
       |  ROB entries: $rob_entries
       |  DMA max bytes: $dma_maxbytes
       |  Mem addr len: $memAddrLen
       |  Bank channel: $bankChannel
       |""".stripMargin
}

@instantiable
class MemDomain(val parameter: MemDomainParam)(edge: TLEdgeOut)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {
    // -------------------------------------------------
    // Command Channel
    // -------------------------------------------------
    // global RS -> MemDomain
    val global_issue_i    = Flipped(Decoupled(new GlobalRsIssue(parameter.rob_entries)(p)))
    // MemDomain -> global RS
    val global_complete_o = Decoupled(new GlobalRsComplete(parameter.rob_entries)(p))
    val busy              = Output(Bool())

    // -------------------------------------------------
    // Inside Channel
    // -------------------------------------------------
    // Bank interface for interaction with Ball Domain
    val ballDomain = new Bundle {

      val bankRead = Vec(
        parameter.bankNum,
        new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.bankNum)
      )

      val bankWrite = Vec(
        parameter.bankNum,
        new BankWrite(
          parameter.bankEntries,
          parameter.bankWidth,
          parameter.bankMaskLen,
          parameter.rob_entries,
          parameter.bankNum
        )
      )

    }

    // -------------------------------------------------
    // Outside Channel
    // -------------------------------------------------
    val dma = new Bundle {

      val read = new Bundle {
        val req  = Decoupled(new BBReadRequest())
        val resp = Flipped(Decoupled(new BBReadResponse(parameter.bankWidth)))
      }

      val write = new Bundle {
        val req  = Decoupled(new BBWriteRequest(parameter.bankWidth))
        val resp = Flipped(Decoupled(new BBWriteResponse))
      }

    }

    val tlb    = Vec(2, Flipped(new BBTLBIO))
    val ptw    = Vec(1, new TLBPTWIO)
    val tlbExp = Vec(1, new BBTLBExceptionIO)
  })

  val frontend: Instance[MemController] = Instantiate(new MemController(parameter)(edge))
  val midend:   Instance[MemScheduler]  = Instantiate(new MemScheduler(parameter))
  val backend:  Instance[MemManager]    = Instantiate(new MemManager(parameter))

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
}
