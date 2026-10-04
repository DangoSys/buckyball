package framework.memdomain.frontend

import chisel3._
import chisel3.util._
import framework.memdomain.frontend.mem.dma.{DmaError, DmaPort, DmaReadCommand}
import framework.memdomain.frontend.mem.{
  Footprint,
  KernelDma,
  KernelDmaPort,
  MemConfiger,
  MemConfigerIO,
  MemLoader,
  MemStorer
}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.frontend.decoder.GISA.MVIN_KERNEL_BITPAT
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd.decoder.MemDomainDecoder
import framework.memdomain.frontend.cmd.rs.MemReservationStation
import framework.memdomain.utils.pmc.MemCyclePMC
import framework.memdomain.isa.{MvoverISA, MvoverPort}

/**
 * MemFrontend:
 * Provides DMA interface and Ball Domain interface
 */
@instantiable
class MemFrontend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Issue interface from global RS (single channel)
    val global_issue_i    = Flipped(Decoupled(new GlobalSchedIssue(b)))
    // Report completion to global RS (single channel)
    val global_complete_o = Decoupled(new GlobalSchedComplete(b))
    val kernel_command    = if (b.rvv.enable) Some(Decoupled(new GlobalSchedIssue(b))) else None
    val kernel_complete   = if (b.rvv.enable) Some(Flipped(Decoupled(new GlobalSchedComplete(b)))) else None
    val mvover            = new MvoverPort

    // Bank read/write interface - used by load/store
    val interdma = new Bundle {
      val bankRead        = Flipped(new BankRead(b))
      val bankWrite       = Flipped(new BankWrite(b))
      val read_is_shared  = Output(Bool())
      val write_is_shared = Output(Bool())
    }

    val dma = new DmaPort(b.memDomain.dma_buswidth)

    val config = Decoupled(new MemConfigerIO(b))

    // MMIO outputs (to MmioPool via MemDomain)
    val is_mvin_mmio_active = Output(Bool())
    val mmio_addr           = Output(UInt(17.W))
    val mmio_col            = Output(UInt(8.W))

    // Query interface to backend for group count
    val query_vbank_id    = Output(UInt(b.memDomain.vbankIdWidth.W))
    val query_is_shared   = Output(Bool())
    val query_group_count = Input(UInt(b.memDomain.groupCountWidth.W))
    val clearBusy         = Input(Bool())

    val hartid     = Input(UInt(b.tile.xLen.W))
    val footprints = Output(Vec(3, new Footprint(b)))

    val kernel = if (b.rvv.enable) Some(new KernelDmaPort) else None

    // Busy signal
    val busy = Output(Bool())
  })

  val memDecoder: Instance[MemDomainDecoder]      = Instantiate(new MemDomainDecoder(b))
  val memRs:      Instance[MemReservationStation] = Instantiate(new MemReservationStation(b))
  val memLoader:  Instance[MemLoader]             = Instantiate(new MemLoader(b))
  val memStorer:  Instance[MemStorer]             = Instantiate(new MemStorer(b))
  val pmc:        Instance[MemCyclePMC]           = Instantiate(new MemCyclePMC(b))

  val configer: Instance[MemConfiger] = Instantiate(new MemConfiger(b))

// -----------------------------------------------------------------------------
// Global RS -> MemDecoder
// -----------------------------------------------------------------------------
  val isMvover       = io.global_issue_i.bits.cmd.cmd.funct === MvoverISA.Funct.U
  val mvoverPending  = RegInit(false.B)
  val mvoverRobId    = Reg(chiselTypeOf(io.global_issue_i.bits.rob_id))
  val mvoverIsSub    = Reg(Bool())
  val mvoverSubRobId = Reg(chiselTypeOf(io.global_issue_i.bits.sub_rob_id))
  val isKernelLoad   = b.rvv.enable.B && io.global_issue_i.bits.cmd.cmd.funct === MVIN_KERNEL_BITPAT
  val kernelPending  = RegInit(false.B)
  val kernelRobId    = Reg(UInt(log2Ceil(b.frontend.rob_entries).W))
  val kernelReady    = if (b.rvv.enable) io.kernel_command.get.ready else false.B
  if (b.rvv.enable) {
    io.kernel_command.get.valid                     := io.global_issue_i.valid && isKernelLoad && !kernelPending && !mvoverPending
    io.kernel_command.get.bits                      := io.global_issue_i.bits
    when(io.kernel_command.get.fire) {
      kernelPending := true.B
      kernelRobId   := io.kernel_command.get.bits.rob_id
    }
    when(io.kernel_complete.get.fire)(kernelPending := false.B)
  }

  io.mvover.command.valid           := io.global_issue_i.valid && isMvover && !mvoverPending && !kernelPending
  io.mvover.command.bits.sourceCore := io.global_issue_i.bits.cmd.cmd.rs1Data(7, 0)
  io.mvover.command.bits.targetCore := io.global_issue_i.bits.cmd.cmd.rs1Data(15, 8)
  io.mvover.command.bits.sourceBank := io.global_issue_i.bits.cmd.cmd.rs1Data(25, 16)
  io.mvover.command.bits.targetBank := io.global_issue_i.bits.cmd.cmd.rs1Data(35, 26)
  io.mvover.command.bits.sourceAddr := io.global_issue_i.bits.cmd.cmd.rs2Data(15, 0)
  io.mvover.command.bits.targetAddr := io.global_issue_i.bits.cmd.cmd.rs2Data(31, 16)
  io.mvover.command.bits.rows       := io.global_issue_i.bits.cmd.cmd.rs2Data(47, 32) +& 1.U
  when(io.mvover.command.fire) {
    mvoverPending  := true.B
    mvoverRobId    := io.global_issue_i.bits.rob_id
    mvoverIsSub    := io.global_issue_i.bits.is_sub
    mvoverSubRobId := io.global_issue_i.bits.sub_rob_id
  }
  memDecoder.io.cmd_i.valid         := io.global_issue_i.valid && !isMvover && !isKernelLoad && !mvoverPending && !kernelPending
  memDecoder.io.cmd_i.bits          := io.global_issue_i.bits.cmd
  io.global_issue_i.ready           := !mvoverPending && !kernelPending &&
    Mux(isKernelLoad, kernelReady, Mux(isMvover, io.mvover.command.ready, memDecoder.io.cmd_i.ready))

  // Config signal goes to backend
  io.config <> configer.io.config

  val selectStorerQuery = memStorer.io.query_valid
  io.query_vbank_id              := Mux(selectStorerQuery, memStorer.io.query_vbank_id, memLoader.io.query_vbank_id)
  io.query_is_shared             := Mux(selectStorerQuery, memStorer.io.query_is_shared, memLoader.io.query_is_shared)
  memLoader.io.query_group_count := io.query_group_count
  memStorer.io.query_group_count := io.query_group_count

  when(
    memLoader.io.query_valid && memStorer.io.query_valid &&
      memLoader.io.query_is_shared && memStorer.io.query_is_shared &&
      memLoader.io.query_vbank_id =/= memStorer.io.query_vbank_id
  ) {
    assert(false.B, "MemFrontend shared query conflict: loader and storer query different vbanks in the same cycle\n")
  }

// -----------------------------------------------------------------------------
// MemDecoder -> MemReservationStation
// -----------------------------------------------------------------------------
  // Connect decoded instruction and global rob_id
  memRs.io.mem_decode_cmd_i.valid           := memDecoder.io.mem_decode_cmd_o.valid
  memRs.io.mem_decode_cmd_i.bits.cmd        := memDecoder.io.mem_decode_cmd_o.bits
  memRs.io.mem_decode_cmd_i.bits.rob_id     := io.global_issue_i.bits.rob_id
  memRs.io.mem_decode_cmd_i.bits.is_sub     := io.global_issue_i.bits.is_sub
  memRs.io.mem_decode_cmd_i.bits.sub_rob_id := io.global_issue_i.bits.sub_rob_id
  memDecoder.io.mem_decode_cmd_o.ready      := memRs.io.mem_decode_cmd_i.ready

// -----------------------------------------------------------------------------
// MemReservationStation -> MemLoader/MemStorer
// -----------------------------------------------------------------------------
  memLoader.io.cmdReq <> memRs.io.issue_o.ld
  memStorer.io.cmdReq <> memRs.io.issue_o.st
  configer.io.cmdReq <> memRs.io.issue_o.cf
  configer.io.hartid    := io.hartid
  configer.io.clearBusy := io.clearBusy
  memRs.io.commit_i.ld <> memLoader.io.cmdResp
  memRs.io.commit_i.st <> memStorer.io.cmdResp
  memRs.io.commit_i.cf <> configer.io.cmdResp

//===-------------------------------------------------------------------===//--
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

  // A logical request is accepted into this queue before its external transport fires.
  // Keep both producer identity and transfer stable while the transport is backpressured.
  val readQueue   = Module(new Queue(new DmaReadCommand, 1))
  val readerOwned = RegInit(false.B)
  val kernelOwner = RegInit(false.B)
  io.dma.read.valid                                                       := readQueue.io.deq.valid && !readerOwned
  io.dma.read.bits                                                        := readQueue.io.deq.bits
  readQueue.io.deq.ready                                                  := io.dma.read.ready && !readerOwned
  when(io.dma.read.fire) {
    readerOwned := true.B
    kernelOwner := io.dma.read.bits.producer === 2.U
  }
  when(io.dma.readResult.fire && io.dma.readResult.bits.last)(readerOwned := false.B)
  when(io.dma.readResult.valid) {
    assert(readerOwned || io.dma.read.fire, "MemFrontend DMA read result has no accepted transport owner")
  }
  io.footprints(0)                                                        := memLoader.io.footprint
  io.footprints(1)                                                        := memStorer.io.footprint
  io.footprints(2)                                                        := 0.U.asTypeOf(new Footprint(b))
  if (b.rvv.enable) {
    val kernelDma = Instantiate(new KernelDma(b))
    kernelDma.io.rob_id := kernelRobId
    io.footprints(2)    := kernelDma.io.footprint
    kernelDma.io.kernel <> io.kernel.get
    val selectKernel = kernelDma.io.request.valid
    readQueue.io.enq.valid         := kernelDma.io.request.valid || memLoader.io.dmaReq.valid
    readQueue.io.enq.bits.transfer := Mux(selectKernel, kernelDma.io.request.bits, memLoader.io.dmaReq.bits)
    readQueue.io.enq.bits.producer := Mux(selectKernel, 2.U, 0.U)
    kernelDma.io.request.ready     := selectKernel && readQueue.io.enq.ready
    memLoader.io.dmaReq.ready      := !selectKernel && readQueue.io.enq.ready
    // KernelDma treats enqueue acceptance as an obligation to drain even after abort.
    kernelDma.io.response.valid    := readerOwned && kernelOwner && io.dma.readResult.valid
    kernelDma.io.response.bits     := io.dma.readResult.bits
    memLoader.io.dmaResp.valid     := readerOwned && !kernelOwner && io.dma.readResult.valid
    memLoader.io.dmaResp.bits      := io.dma.readResult.bits
    io.dma.readResult.ready        := readerOwned && Mux(kernelOwner, kernelDma.io.response.ready, memLoader.io.dmaResp.ready)
  } else {
    readQueue.io.enq.valid         := memLoader.io.dmaReq.valid
    readQueue.io.enq.bits.transfer := memLoader.io.dmaReq.bits
    readQueue.io.enq.bits.producer := 0.U
    memLoader.io.dmaReq.ready      := readQueue.io.enq.ready
    memLoader.io.dmaResp.valid     := readerOwned && io.dma.readResult.valid
    memLoader.io.dmaResp.bits      := io.dma.readResult.bits
    io.dma.readResult.ready        := readerOwned && memLoader.io.dmaResp.ready
  }
  io.dma.write <> memStorer.io.dmaReq
  io.dma.writeData <> memStorer.io.dmaData
  memStorer.io.dmaResp <> io.dma.writeResult

  // MemLoader owns the frontend write channel until its write response returns,
  // preserving BankWrite's request/response pairing.
  val frontendWriteBusy = RegInit(false.B)

  io.interdma.bankWrite.io.req.valid  := !frontendWriteBusy && memLoader.io.bankWrite.io.req.valid
  io.interdma.bankWrite.io.req.bits   := memLoader.io.bankWrite.io.req.bits
  io.interdma.bankWrite.bank_id       := memLoader.io.bankWrite.bank_id
  io.interdma.bankWrite.rob_id        := memLoader.io.bankWrite.rob_id
  io.interdma.bankWrite.ball_id       := memLoader.io.bankWrite.ball_id
  io.interdma.bankWrite.group_id      := memLoader.io.bankWrite.group_id
  memLoader.io.bankWrite.io.req.ready := !frontendWriteBusy && io.interdma.bankWrite.io.req.ready

  io.interdma.bankWrite.io.resp.ready  := frontendWriteBusy && memLoader.io.bankWrite.io.resp.ready
  memLoader.io.bankWrite.io.resp.valid := frontendWriteBusy && io.interdma.bankWrite.io.resp.valid
  memLoader.io.bankWrite.io.resp.bits  := io.interdma.bankWrite.io.resp.bits

  when(io.interdma.bankWrite.io.req.fire)(frontendWriteBusy  := true.B)
  when(io.interdma.bankWrite.io.resp.fire)(frontendWriteBusy := false.B)

  memStorer.io.bankRead <> io.interdma.bankRead
  io.interdma.read_is_shared  := memStorer.io.is_shared
  io.interdma.write_is_shared := memLoader.io.is_shared

  // MMIO signals from MemConfiger and MemLoader, exposed to MemDomain
  io.is_mvin_mmio_active := memLoader.io.is_mvin_mmio_active
  io.mmio_addr           := memLoader.io.mmio_addr
  io.mmio_col            := memLoader.io.mmio_col

  // Completion signal connected to global RS
  val mvoverComplete = mvoverPending && io.mvover.completion.valid
  val kernelComplete = if (b.rvv.enable) kernelPending && io.kernel_complete.get.valid else false.B
  io.global_complete_o.valid           := kernelComplete || mvoverComplete || memRs.io.complete_o.valid
  io.global_complete_o.bits.rob_id     := Mux(mvoverComplete, mvoverRobId, memRs.io.complete_o.bits.rob_id)
  io.global_complete_o.bits.is_sub     := Mux(mvoverComplete, mvoverIsSub, memRs.io.complete_o.bits.is_sub)
  io.global_complete_o.bits.sub_rob_id := Mux(mvoverComplete, mvoverSubRobId, memRs.io.complete_o.bits.sub_rob_id)
  io.global_complete_o.bits.fault      := Mux(
    mvoverComplete,
    0.U.asTypeOf(io.global_complete_o.bits.fault),
    memRs.io.complete_o.bits.fault
  )
  when(mvoverComplete && io.mvover.completion.bits) {
    io.global_complete_o.bits.fault.error := DmaError.Bank.U
  }
  if (b.rvv.enable) {
    when(kernelComplete)(io.global_complete_o.bits := io.kernel_complete.get.bits)
    io.kernel_complete.get.ready                   := kernelPending && io.global_complete_o.ready
  }
  memRs.io.complete_o.ready            := io.global_complete_o.ready && !mvoverComplete && !kernelComplete
  io.mvover.completion.ready           := io.global_complete_o.ready && mvoverPending && !kernelComplete
  when(io.mvover.completion.fire) {
    mvoverPending := false.B
  }

  // Busy signal
  // Simple busy signal
  io.busy := kernelPending || mvoverPending || !memRs.io.complete_o.ready || io.kernel.map(_.busy).getOrElse(false.B) ||
    readQueue.io.deq.valid || readerOwned || io.dma.readBusy || io.dma.writeBusy
}
