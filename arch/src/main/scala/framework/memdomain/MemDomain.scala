package framework.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tile._
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.balldomain.blink.mmio.MmioRead
import freechips.rocketchip.tilelink.{TLBundle, TLEdgeOut}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.top.GlobalConfig
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.mmio.MmioPool
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.MemFrontend
import framework.memdomain.frontend.mem.{MemConfigerIO}
import framework.memdomain.frontend.mem.tlb.{BBTLBExceptionIO, BBTLBPTWIO}
import framework.memdomain.midend.MemMidend
import framework.memdomain.backend.MemBackend

@instantiable
class MemDomain(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
// Command Channel
    val global_issue_i    = Flipped(Decoupled(new GlobalSchedIssue(b)))
    val global_complete_o = Decoupled(new GlobalSchedComplete(b))
    val busy              = Output(Bool())

// Inside Channel
    val ballDomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))
      val mmioRead  = Vec(b.ballDomain.ballNum, new MmioRead(b))
    }

// Outside Channel
    val ptw       = Vec(1, new BBTLBPTWIO(b))
    val tlbExp    = Vec(1, new BBTLBExceptionIO)
    val tl_reader = new TLBundle(edge.bundle)
    val tl_writer = new TLBundle(edge.bundle)
    val hartid    = Input(UInt(b.core.xLen.W))

// Shared memory path
    val shared_mem_req           = Vec(SharedMemLayout.channelPerHart(b), new MemRequestIO(b))
    val shared_config            = Decoupled(new MemConfigerIO(b))
    val shared_query_vbank_id    = Output(UInt(8.W))
    val shared_query_group_count = Input(UInt(4.W))
  })

  val frontend: Instance[MemFrontend] = Instantiate(new MemFrontend(b)(edge))
  val midend:   Instance[MemMidend]   = Instantiate(new MemMidend(b))
  val backend:  Instance[MemBackend]  = Instantiate(new MemBackend(b))
  val mmioPool: Instance[MmioPool]    = Instantiate(new MmioPool(b))

  // Connect query interface from frontend to backend
  backend.io.query_vbank_id     := frontend.io.query_vbank_id
  backend.io.query_is_shared    := frontend.io.query_is_shared
  frontend.io.query_group_count := backend.io.query_group_count
  frontend.io.hartid            := io.hartid

  // Shared query: backend delegates shared query to external SharedMemBackend
  backend.io.shared_query_group_count := io.shared_query_group_count
  io.shared_query_vbank_id            := backend.io.shared_query_vbank_id

//===----------------------------------------------------------------------===//
// Connection with outside (all in frontend)
//===----------------------------------------------------------------------===//
  frontend.io.global_issue_i <> io.global_issue_i
  frontend.io.global_complete_o <> io.global_complete_o
  io.busy := frontend.io.busy

  frontend.io.ptw <> io.ptw
  frontend.io.tlbExp <> io.tlbExp

  io.tl_reader <> frontend.io.tl_reader
  io.tl_writer <> frontend.io.tl_writer

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

//===----------------------------------------------------------------------===//
// MMIO subsystem wiring
//===----------------------------------------------------------------------===//
  // Alloc/dealloc from MemConfiger (mmio_set instruction)
  mmioPool.io.alloc := frontend.io.mmioAlloc

  // Dealloc from MemConfiger (implicit lifecycle: mem_dealloc with alloc=false)
  // For now, tie off explicit dealloc (lifecycle managed by mmio_set with size_rows=0)
  mmioPool.io.dealloc.valid := false.B
  mmioPool.io.dealloc.bits  := 0.U

  // Write path: route MemLoader's bankWrite to MmioPool when is_mvin_mmio_active
  val loaderBankWrite = frontend.io.interdma.bankWrite
  val destIsMmio      = frontend.io.is_mvin_mmio_active

  // Compute MMIO write parameters from MemLoader's mmio_addr/col
  val mmioRowAddr  = frontend.io.mmio_addr(9, 4) + loaderBankWrite.io.req.bits.addr
  val mmioBankIdx  = frontend.io.mmio_addr(16, 10)
  val mmioByteMask = Wire(Vec(b.memDomain.bankMaskLen, Bool()))
  for (k <- 0 until b.memDomain.bankMaskLen) {
    mmioByteMask(k) := k.U < frontend.io.mmio_col
  }

  // Route write to MMIO or main bank based on is_mvin_mmio_active
  mmioPool.io.write.req.valid      := loaderBankWrite.io.req.valid && destIsMmio
  mmioPool.io.write.req.bits.addr  := mmioRowAddr
  mmioPool.io.write.req.bits.data  := loaderBankWrite.io.req.bits.data
  mmioPool.io.write.req.bits.mask  := mmioByteMask
  mmioPool.io.write.req.bits.wmode := false.B
  mmioPool.io.writeBankIdx         := mmioBankIdx

  // Main bank write (when NOT mvin_mmio)
  midend.io.bankWrite(totalBallWrite).bankWrite.io.req.valid := loaderBankWrite.io.req.valid && !destIsMmio
  midend.io.bankWrite(totalBallWrite).bankWrite.io.req.bits  := loaderBankWrite.io.req.bits
  midend.io.bankWrite(totalBallWrite).bankWrite.rob_id       := loaderBankWrite.rob_id
  midend.io.bankWrite(totalBallWrite).bankWrite.bank_id      := loaderBankWrite.bank_id
  midend.io.bankWrite(totalBallWrite).bankWrite.ball_id      := loaderBankWrite.ball_id
  midend.io.bankWrite(totalBallWrite).bankWrite.group_id     := loaderBankWrite.group_id
  midend.io.bankWrite(totalBallWrite).is_shared              := frontend.io.interdma.write_is_shared

  // Request ready mux: select MMIO or main bank ready
  loaderBankWrite.io.req.ready := Mux(
    destIsMmio,
    mmioPool.io.write.req.ready,
    midend.io.bankWrite(totalBallWrite).bankWrite.io.req.ready
  )

  // Response mux: select MMIO or main bank response
  loaderBankWrite.io.resp.valid := Mux(
    destIsMmio,
    mmioPool.io.write.resp.valid,
    midend.io.bankWrite(totalBallWrite).bankWrite.io.resp.valid
  )
  loaderBankWrite.io.resp.bits  := Mux(
    destIsMmio,
    mmioPool.io.write.resp.bits,
    midend.io.bankWrite(totalBallWrite).bankWrite.io.resp.bits
  )

  // Ready signals
  mmioPool.io.write.resp.ready                                := loaderBankWrite.io.resp.ready && destIsMmio
  midend.io.bankWrite(totalBallWrite).bankWrite.io.resp.ready := loaderBankWrite.io.resp.ready && !destIsMmio

  // Ball read path: connect Ball mmioRead to MmioPool
  for (i <- 0 until b.ballDomain.ballNum) {
    mmioPool.io.ballReq(i) <> io.ballDomain.mmioRead(i).req
    io.ballDomain.mmioRead(i).resp <> mmioPool.io.ballResp(i)
    mmioPool.io.ballMetaBank(i) := io.ballDomain.mmioRead(i).meta_bank
  }

  // Shared path passthrough
  io.shared_mem_req <> backend.io.shared_mem_req
  io.shared_config <> backend.io.shared_config
}
