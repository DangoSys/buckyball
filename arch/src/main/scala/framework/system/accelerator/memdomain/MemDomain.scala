package framework.system.accelerator.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tile._
import framework.balldomain.blink.{BankRead, BankWrite}
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.top.GlobalConfig
import framework.system.link.BBAxi4Params
import framework.system.accelerator.memdomain.backend.MemRequestIO
import framework.system.accelerator.memdomain.frontend.MemFrontend
import framework.system.accelerator.memdomain.frontend.outside_channel.{MemConfigerIO}
import framework.system.accelerator.memdomain.frontend.outside_channel.tlb.{BBTLBExceptionIO, BBTLBPTWIO}
import framework.system.accelerator.memdomain.midend.MemMidend
import framework.system.accelerator.memdomain.backend.MemBackend

@instantiable
class MemDomain(val b: GlobalConfig) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  val axiParams: AXI4BundleParameters = BBAxi4Params(b)

  @public
  val io = IO(new Bundle {
    // -------------------------------------------------
    // Command Channel
    // -------------------------------------------------
    val global_issue_i    = Flipped(Decoupled(new GlobalSchedIssue(b)))
    val global_complete_o = Decoupled(new GlobalSchedComplete(b))
    val busy              = Output(Bool())

    // -------------------------------------------------
    // Inside Channel
    // -------------------------------------------------
    val ballDomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))
    }

    // -------------------------------------------------
    // Outside Channel
    // -------------------------------------------------
    val ptw        = Vec(1, new BBTLBPTWIO(b))
    val tlbExp     = Vec(1, new BBTLBExceptionIO)
    val axi_reader = AXI4Bundle(axiParams)
    val axi_writer = AXI4Bundle(axiParams)
    val hartid     = Input(UInt(b.core.xLen.W))

    // -------------------------------------------------
    // Shared memory path — exposed for tile-level sharing
    // -------------------------------------------------
    val shared_mem_req           = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
    val shared_config            = Decoupled(new MemConfigerIO(b))
    val shared_query_vbank_id    = Output(UInt(8.W))
    val shared_query_group_count = Input(UInt(4.W))
  })

  val frontend: Instance[MemFrontend] = Instantiate(new MemFrontend(b))
  val midend:   Instance[MemMidend]   = Instantiate(new MemMidend(b))
  val backend:  Instance[MemBackend]  = Instantiate(new MemBackend(b))

  // Connect query interface from frontend to backend
  backend.io.query_vbank_id     := frontend.io.query_vbank_id
  backend.io.query_is_shared    := frontend.io.query_is_shared
  frontend.io.query_group_count := backend.io.query_group_count
  frontend.io.hartid            := io.hartid

  // Shared query: backend delegates shared query to external SharedMemBackend
  backend.io.shared_query_group_count := io.shared_query_group_count
  io.shared_query_vbank_id            := backend.io.shared_query_vbank_id

  // -------------------------------------------------
  // Connection with outside (all in frontend)
  // -------------------------------------------------
  frontend.io.global_issue_i <> io.global_issue_i
  frontend.io.global_complete_o <> io.global_complete_o
  io.busy := frontend.io.busy

  frontend.io.ptw <> io.ptw
  frontend.io.tlbExp <> io.tlbExp

  io.axi_reader <> frontend.io.axi_reader
  io.axi_writer <> frontend.io.axi_writer

  // Ball Domain interface connects to midend unified bankRead/bankWrite
  // Indices [0, totalBallRead) are balldomain; last index is frontend (DMA)
  for (i <- 0 until totalBallRead) {
    midend.io.bankRead(i).bankRead <> io.ballDomain.bankRead(i)
    midend.io.bankRead(i).is_shared := false.B
  }
  for (i <- 0 until totalBallWrite) {
    midend.io.bankWrite(i).bankWrite <> io.ballDomain.bankWrite(i)
    midend.io.bankWrite(i).is_shared := false.B
  }
  midend.io.bankRead(totalBallRead).bankRead <> frontend.io.interdma.bankRead
  midend.io.bankRead(totalBallRead).is_shared := frontend.io.interdma.read_is_shared
  midend.io.bankWrite(totalBallWrite).bankWrite <> frontend.io.interdma.bankWrite
  midend.io.bankWrite(totalBallWrite).is_shared := frontend.io.interdma.write_is_shared
  midend.io.hartid                              := io.hartid

  midend.io.mem_req <> backend.io.mem_req
  backend.io.config <> frontend.io.config

  // Shared path passthrough
  io.shared_mem_req <> backend.io.shared_mem_req
  io.shared_config <> backend.io.shared_config
}
