package framework.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tile._
import framework.balldomain.blink.{BankRead, BankWrite}
import freechips.rocketchip.tilelink.{TLBundle, TLEdgeOut}
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.top.GlobalConfig

import framework.memdomain.frontend.MemFrontend
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBExceptionIO, BBTLBPTWIO}
import framework.memdomain.midend.MemMidend
import framework.memdomain.backend.MemBackend

@instantiable
class MemDomain(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

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
    // BankRead/BankWrite are used with Flipped at Ball Device side
    // MemDomain receives requests from Ball Domain, so uses raw Bundle (Input for bank_id)
    // Use bbusProducerChannels and bbusConsumerChannels instead of bankNum to match BallDomain output
    val ballDomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))

    }

    // -------------------------------------------------
    // Outside Channel
    // -------------------------------------------------
    // PTW and TLB exception interfaces for external connection
    val ptw    = Vec(1, new BBTLBPTWIO(b))
    val tlbExp = Vec(1, new BBTLBExceptionIO)

    // TileLink physical connections for DMA
    val tl_reader = new TLBundle(edge.bundle)
    val tl_writer = new TLBundle(edge.bundle)
  })

  val frontend: Instance[MemFrontend] = Instantiate(new MemFrontend(b)(edge))
  val midend:   Instance[MemMidend]   = Instantiate(new MemMidend(b))
  val backend:  Instance[MemBackend]  = Instantiate(new MemBackend(b))

  // -------------------------------------------------
  // Connection with outside (all in frontend)
  // -------------------------------------------------
  // Global RS interface
  frontend.io.global_issue_i <> io.global_issue_i
  frontend.io.global_complete_o <> io.global_complete_o
  io.busy := frontend.io.busy

  // TLB interface
  // Note: frontend.io.tlb is connected internally to DMA modules
  // Only PTW and TLB exception interfaces are exposed to outside
  frontend.io.ptw <> io.ptw
  frontend.io.tlbExp <> io.tlbExp

  // TileLink physical connections for DMA (Reader/Writer are inside frontend)
  io.tl_reader <> frontend.io.tl_reader
  io.tl_writer <> frontend.io.tl_writer

  // Ball Domain interface connects directly to midend (Ball devices' read/write requests)
  midend.io.balldomain.bankRead <> io.ballDomain.bankRead
  midend.io.balldomain.bankWrite <> io.ballDomain.bankWrite
  midend.io.frontend.bankRead <> frontend.io.interdma.bankRead
  midend.io.frontend.bankWrite <> frontend.io.interdma.bankWrite

  midend.io.mem_req <> backend.io.mem_req
  backend.io.acc_config <> frontend.io.acc_config
}
