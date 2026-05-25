package framework.memdomain.backend.mmio

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.banks.SramWriteIO
import framework.top.GlobalConfig

/**
 * MmioPool: top-level MMIO subsystem wrapper.
 *
 * Instantiates:
 *   - Vec of MmioBanks (one per mmioBankNum)
 *   - MmioRegionTable (binding table)
 *   - MmioRouter (translates Ball requests to bank requests)
 *
 * External interfaces:
 *   - alloc/dealloc: from MemConfiger (mmio_set / mem_dealloc instructions)
 *   - write/writeBankIdx: from MemLoader (mvin_mmio DMA path)
 *   - ballReq/ballResp/ballMetaBank: from Balls (via MemDomain wiring)
 */
@instantiable
class MmioPool(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Region table management (from MemConfiger)
    val alloc   = Flipped(Valid(new MmioAllocReq(b)))
    val dealloc = Flipped(Valid(UInt(log2Up(b.memDomain.bankNum).W)))

    // DMA write path (from MemLoader)
    val write        = new SramWriteIO(b)
    val writeBankIdx = Input(UInt(log2Ceil(b.memDomain.mmioBankNum).W))

    // Ball read path (from MemDomain, one port per Ball)
    val ballReq      =
      Vec(b.ballDomain.ballNum, Flipped(Decoupled(new MmioReadReq(b))))
    val ballResp     = Vec(b.ballDomain.ballNum, Decoupled(new MmioReadResp(b)))
    val ballMetaBank =
      Input(Vec(b.ballDomain.ballNum, UInt(log2Up(b.memDomain.bankNum).W)))
  })

  // Instantiate components
  val banks  = Seq.fill(b.memDomain.mmioBankNum)(Instantiate(new MmioBank(b)))
  val table  = Instantiate(new MmioRegionTable(b))
  val router = Instantiate(new MmioRouter(b))

  // Wire region table
  table.io.alloc   := io.alloc
  table.io.dealloc := io.dealloc

  // Wire router to table and Balls
  router.io.ballReq <> io.ballReq
  io.ballResp <> router.io.ballResp
  router.io.ballMetaBank := io.ballMetaBank

  table.io.lookup_main_bank   := router.io.tableLookupBank
  router.io.tableLookupResult := table.io.lookup_result

  // Wire router to banks (read path)
  for (i <- 0 until b.memDomain.mmioBankNum) {
    banks(i).io.read.req <> router.io.bankReadReq(i)
    banks(i).io.read.resp <> router.io.bankReadResp(i)
  }

  // Wire DMA write path: demux io.write to banks(writeBankIdx)
  for (i <- 0 until b.memDomain.mmioBankNum) {
    banks(
      i
    ).io.write.req.valid       := io.write.req.valid && (io.writeBankIdx === i.U)
    banks(i).io.write.req.bits := io.write.req.bits
    // resp.ready needs to be driven so the bank knows the consumer accepts responses
    banks(
      i
    ).io.write.resp.ready      := io.write.resp.ready && (io.writeBankIdx === i.U)
  }

  val writeRespMux = Mux1H(
    UIntToOH(io.writeBankIdx, b.memDomain.mmioBankNum),
    banks.map(_.io.write.resp)
  )

  io.write.resp.valid := writeRespMux.valid
  io.write.resp.bits  := writeRespMux.bits

  val writeReadyMux = Mux1H(
    UIntToOH(io.writeBankIdx, b.memDomain.mmioBankNum),
    banks.map(_.io.write.req.ready)
  )

  io.write.req.ready := writeReadyMux
}
