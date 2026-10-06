package framework.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.top.GlobalConfig
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.mmio.MmioPool
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.MemFrontend
import framework.memdomain.frontend.mem.{Footprint, KernelMemoryBridge, MemConfigerIO}
import framework.memdomain.frontend.mem.dma.DmaPort
import framework.memdomain.midend.MemMidend
import framework.memdomain.backend.MemBackend
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.isa.{MvoverISA, MvoverPort}
import memcore.memory.mesh_shm.MeshLocalBankPort

@instantiable
class MemDomain(val b: GlobalConfig) extends Module {
  val totalMmioRead   = b.ballDomain.ballIdMappings.map(_.mmioReadBW).sum
  val totalMmioWrite  = b.ballDomain.ballIdMappings.map(_.mmioWriteBW).sum
  val totalBallRead   = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite  = b.ballDomain.ballIdMappings.map(_.outBW).sum
  val kernelPorts     = if (b.rvv.enable) b.rvv.memoryPorts else 0
  val sharedHashCount = if (b.memDomain.sharedEnable) SharedMemLayout.totalBank(b) else 0

  @public
  val io = IO(new Bundle {
// Command Channel
    val global_issue_i    = Flipped(Decoupled(new GlobalSchedIssue(b)))
    val global_complete_o = Decoupled(new GlobalSchedComplete(b))
    val kernel_command    = if (b.rvv.enable) Some(Decoupled(new GlobalSchedIssue(b))) else None
    val kernel_complete   = if (b.rvv.enable) Some(Flipped(Decoupled(new GlobalSchedComplete(b)))) else None
    val busy              = Output(Bool())
    val inst_ids          = Input(Vec(b.frontend.rob_entries, UInt(64.W)))
    val footprints        = Output(Vec(3, new Footprint(b)))

// Inside Channel
    val ballChannelActive = Input(Vec(b.ballDomain.ballNum, Bool()))
    val ballChannelReady  = Output(Vec(b.ballDomain.ballNum, Bool()))

    val ballDomain = new Bundle {
      val bankRead  = Vec(totalBallRead, new BankRead(b))
      val bankWrite = Vec(totalBallWrite, new BankWrite(b))
      val mmioRead  = Vec(totalMmioRead, new MmioRead(b))
      val mmioWrite = Vec(totalMmioWrite, new MmioWrite(b))
    }

    val kernel =
      if (b.rvv.enable) Some(new KernelMemoryBridge(b))
      else None

// Outside Channel
    val dma    = new DmaPort(b.memDomain.dma_buswidth)
    val hartid = Input(UInt(b.tile.xLen.W))

// Shared memory path
    val shared_mem_req = Vec(SharedMemLayout.channelPerHart(b), new MemRequestIO(b))
    val mvover         = new MvoverPort

    val meshLocalBank = Flipped(new MeshLocalBankPort(
      MvoverISA.AddressBits,
      MvoverISA.BankBits,
      b.memDomain.bankWidth,
      math.max(1, log2Ceil(b.frontend.rob_entries))
    ))

    val shared_config            = Decoupled(new MemConfigerIO(b))
    val shared_query_valid       = Output(Bool())
    val shared_query_vbank_id    = Output(UInt(b.memDomain.vbankIdWidth.W))
    val shared_query_group_count = Input(UInt(b.memDomain.groupCountWidth.W))

    val bank_hashes =
      if (b.sim.diffTest) {
        Some(Output(Vec(b.memDomain.bankNum + sharedHashCount, new PhysicalBankHash(b))))
      } else {
        None
      }

    val shared_bank_hashes =
      if (b.sim.diffTest && b.memDomain.sharedEnable) {
        Some(Input(Vec(sharedHashCount, new PhysicalBankHash(b))))
      } else {
        None
      }

  })

  val frontend: Instance[MemFrontend] = Instantiate(new MemFrontend(b))
  val midend:   Instance[MemMidend]   = Instantiate(new MemMidend(b))
  val backend:  Instance[MemBackend]  = Instantiate(new MemBackend(b))

  io.bank_hashes.foreach(_                := backend.io.bank_hashes.get)
  backend.io.shared_bank_hashes.foreach(_ := io.shared_bank_hashes.get)

  // Connect query interface from frontend to backend
  backend.io.query_vbank_id     := frontend.io.query_vbank_id
  backend.io.query_is_shared    := frontend.io.query_is_shared
  frontend.io.query_group_count := backend.io.query_group_count
  frontend.io.clearBusy         := backend.io.clearBusy
  frontend.io.hartid            := io.hartid
  midend.io.inst_ids            := io.inst_ids

  // Shared query: backend delegates shared query to external SharedMemBackend
  backend.io.shared_query_group_count := io.shared_query_group_count
  io.shared_query_valid               := backend.io.shared_query_valid
  io.shared_query_vbank_id            := backend.io.shared_query_vbank_id

//===----------------------------------------------------------------------===//
// Connection with outside (all in frontend)
//===----------------------------------------------------------------------===//
  frontend.io.global_issue_i <> io.global_issue_i
  frontend.io.global_complete_o <> io.global_complete_o
  if (b.rvv.enable) {
    io.kernel_command.get <> frontend.io.kernel_command.get
    frontend.io.kernel_complete.get <> io.kernel_complete.get
  }
  io.mvover <> frontend.io.mvover
  io.busy       := frontend.io.busy
  io.footprints := frontend.io.footprints

  io.dma <> frontend.io.dma

  // Ball Domain interface connects to midend unified bankRead/bankWrite
  // Indices [0, totalBallRead) are balldomain; last index is frontend (DMA)
  for (i <- 0 until totalBallRead) {
    midend.io.bankRead(i).bankRead <> io.ballDomain.bankRead(i)
    midend.io.bankRead(i).is_shared := false.B
  }

  midend.io.ballChannelActive := io.ballChannelActive
  io.ballChannelReady         := midend.io.ballChannelReady

  for (i <- 0 until totalBallWrite) {
    midend.io.bankWrite(i).bankWrite <> io.ballDomain.bankWrite(i)
    midend.io.bankWrite(i).is_shared := false.B
  }

  if (b.rvv.enable) {
    val kernel = io.kernel.get
    val dma    = frontend.io.kernel.get
    dma.load <> kernel.load
    kernel.image <> dma.image
    kernel.result <> dma.result
    dma.abort    := kernel.abort
    kernel.busy  := dma.busy
    kernel.ready := kernel.active
    for (port <- 0 until kernelPorts) {
      val read = backend.io.kernel_req(port)
      read.read <> kernel.bankRead(port).io
      read.write.req.valid  := false.B
      read.write.req.bits   := 0.U.asTypeOf(read.write.req.bits)
      read.write.resp.ready := true.B
      read.bank_id          := kernel.bankRead(port).bank_id
      read.group_id         := kernel.bankRead(port).group_id
      read.rob_id           := kernel.bankRead(port).rob_id
      read.inst_id          := io.inst_ids(kernel.bankRead(port).rob_id)
      read.hart_id          := io.hartid
      read.is_shared        := false.B

      val write = backend.io.kernel_req(kernelPorts + port)
      write.write <> kernel.bankWrite(port).io
      write.read.req.valid  := false.B
      write.read.req.bits   := 0.U.asTypeOf(write.read.req.bits)
      write.read.resp.ready := true.B
      write.bank_id         := kernel.bankWrite(port).bank_id
      write.group_id        := kernel.bankWrite(port).group_id
      write.rob_id          := kernel.bankWrite(port).rob_id
      write.inst_id         := io.inst_ids(kernel.bankWrite(port).rob_id)
      write.hart_id         := io.hartid
      write.is_shared       := false.B
    }
  }

  midend.io.bankRead(totalBallRead).bankRead <> frontend.io.interdma.bankRead
  midend.io.bankRead(totalBallRead).is_shared := frontend.io.interdma.read_is_shared
  midend.io.hartid                            := io.hartid

  for (i <- 1 until b.memDomain.bankChannel) {
    midend.io.mem_req(i) <> backend.io.mem_req(i)
  }
  if (b.memDomain.sharedEnable && b.memDomain.bankWidth == 128) {
    // Reserve channel 0 for Mesh-initiated private-Bank access only when its
    // ordinary request/response pair has drained. This preserves the normal
    // AccPipe route and prevents one-cycle SRAM responses from being lost.
    val normal             = midend.io.mem_req(0)
    val routed             = backend.io.mem_req(0)
    val local              = io.meshLocalBank
    val normalReadPending  = RegInit(false.B)
    val normalWritePending = RegInit(false.B)
    val localPending       = RegInit(false.B)
    val localIsWrite       = RegInit(false.B)
    val localTag           = Reg(chiselTypeOf(local.request.bits.tag))
    val normalActive       = normalReadPending || normalWritePending ||
      normal.read.req.valid || normal.write.req.valid
    val localRequest       = local.request.valid && !localPending && !normalActive
    val localRead          = localRequest && !local.request.bits.write
    val localWrite         = localRequest && local.request.bits.write

    routed.bank_id             := Mux(localRequest, local.request.bits.bank, normal.bank_id)
    routed.group_id            := Mux(localRequest, 0.U, normal.group_id)
    routed.is_shared           := Mux(localRequest, false.B, normal.is_shared)
    routed.hart_id             := Mux(localRequest, io.hartid, normal.hart_id)
    routed.rob_id              := Mux(localRequest, 0.U, normal.rob_id)
    routed.inst_id             := Mux(localRequest, 0.U, normal.inst_id)
    routed.read.req.valid      := localRead || (normal.read.req.valid && !localPending)
    routed.read.req.bits.addr  := Mux(localRead, local.request.bits.addr, normal.read.req.bits.addr)
    routed.write.req.valid     := localWrite || (normal.write.req.valid && !localPending)
    routed.write.req.bits.addr := Mux(localWrite, local.request.bits.addr, normal.write.req.bits.addr)
    routed.write.req.bits.data := Mux(localWrite, local.request.bits.data, normal.write.req.bits.data)
    routed.write.req.bits.mask := Mux(localWrite, VecInit(local.request.bits.mask.asBools), normal.write.req.bits.mask)
    normal.read.req.ready      := !localPending && routed.read.req.ready
    normal.write.req.ready     := !localPending && routed.write.req.ready
    local.request.ready        := !normalActive && !localPending &&
      Mux(local.request.bits.write, routed.write.req.ready, routed.read.req.ready)

    routed.read.resp.ready    := Mux(localPending && !localIsWrite, local.response.ready, normal.read.resp.ready)
    routed.write.resp.ready   := Mux(localPending && localIsWrite, local.response.ready, normal.write.resp.ready)
    normal.read.resp.valid    := routed.read.resp.valid && !localPending
    normal.read.resp.bits     := routed.read.resp.bits
    normal.write.resp.valid   := routed.write.resp.valid && !localPending
    normal.write.resp.bits    := routed.write.resp.bits
    local.response.valid      := localPending && Mux(localIsWrite, routed.write.resp.valid, routed.read.resp.valid)
    local.response.bits.data  := routed.read.resp.bits.data
    local.response.bits.tag   := localTag
    local.response.bits.error := false.B

    when(normal.read.req.fire)(normalReadPending    := true.B)
    when(normal.read.resp.fire)(normalReadPending   := false.B)
    when(normal.write.req.fire)(normalWritePending  := true.B)
    when(normal.write.resp.fire)(normalWritePending := false.B)
    when(local.request.fire) {
      assert(local.request.bits.bank <= b.frontend.vbank_id_upper_bound.U)
      assert(local.request.bits.addr < b.memDomain.bankEntries.U)
      assert(!local.request.bits.write || local.request.bits.mask.andR)
      localPending := true.B
      localIsWrite := local.request.bits.write
      localTag     := local.request.bits.tag
    }
    when(local.response.fire)(localPending          := false.B)
  } else {
    midend.io.mem_req(0) <> backend.io.mem_req(0)
    io.meshLocalBank.request.ready  := false.B
    io.meshLocalBank.response.valid := false.B
    io.meshLocalBank.response.bits  := 0.U.asTypeOf(io.meshLocalBank.response.bits)
  }
  backend.io.config <> frontend.io.config

//===----------------------------------------------------------------------===//
// MMIO subsystem wiring
//===----------------------------------------------------------------------===//
  val loaderBankWrite = frontend.io.interdma.bankWrite
  val dmaBankWrite    = midend.io.bankWrite(totalBallWrite).bankWrite
  midend.io.bankWrite(totalBallWrite).is_shared := frontend.io.interdma.write_is_shared

  if (b.memDomain.mmioEnable) {
    val mmioPool: Instance[MmioPool] = Instantiate(new MmioPool(b))

    // Write path: route MemLoader's bankWrite to MmioPool when is_mvin_mmio_active
    val destIsMmio = frontend.io.is_mvin_mmio_active

    // MMIO is one globally encoded byte space. Each DMA beat is 16 bytes and
    // MmioPool stripes those bytes across the five physical byte banks.
    val mmioWriteAddr = frontend.io.mmio_addr + loaderBankWrite.io.req.bits.addr * (b.memDomain.bankWidth / 8).U
    val mmioByteMask  = Wire(Vec(b.memDomain.bankMaskLen, Bool()))
    for (k <- 0 until b.memDomain.bankMaskLen) {
      mmioByteMask(k) := k.U < frontend.io.mmio_col
    }

    dmaBankWrite.bank_id  := loaderBankWrite.bank_id
    dmaBankWrite.rob_id   := loaderBankWrite.rob_id
    dmaBankWrite.ball_id  := loaderBankWrite.ball_id
    dmaBankWrite.group_id := loaderBankWrite.group_id

    // Route write to MMIO or main bank based on is_mvin_mmio_active
    mmioPool.io.write.req.valid     := loaderBankWrite.io.req.valid && destIsMmio
    mmioPool.io.write.req.bits.addr := loaderBankWrite.io.req.bits.addr
    mmioPool.io.write.req.bits.data := loaderBankWrite.io.req.bits.data
    mmioPool.io.write.req.bits.mask := mmioByteMask
    mmioPool.io.writeAddr           := mmioWriteAddr

    // Main bank write (when NOT mvin_mmio).
    dmaBankWrite.io.req.valid := loaderBankWrite.io.req.valid && !destIsMmio
    dmaBankWrite.io.req.bits  := loaderBankWrite.io.req.bits

    // Request ready mux: select MMIO or main bank ready
    loaderBankWrite.io.req.ready := Mux(
      destIsMmio,
      mmioPool.io.write.req.ready,
      dmaBankWrite.io.req.ready
    )

    // Response mux: select MMIO or main bank response
    loaderBankWrite.io.resp.valid := Mux(
      destIsMmio,
      mmioPool.io.write.resp.valid,
      dmaBankWrite.io.resp.valid
    )
    loaderBankWrite.io.resp.bits  := Mux(
      destIsMmio,
      mmioPool.io.write.resp.bits,
      dmaBankWrite.io.resp.bits
    )

    // Ready signals
    mmioPool.io.write.resp.ready := loaderBankWrite.io.resp.ready && destIsMmio
    dmaBankWrite.io.resp.ready   := loaderBankWrite.io.resp.ready && !destIsMmio

    // Ball read path: connect every configured Blink MMIO line to MmioPool.
    for (i <- 0 until totalMmioRead) {
      mmioPool.io.ballReq(i) <> io.ballDomain.mmioRead(i).req
      io.ballDomain.mmioRead(i).resp <> mmioPool.io.ballResp(i)
    }
    for (i <- 0 until totalMmioWrite) {
      mmioPool.io.ballWriteReq(i) <> io.ballDomain.mmioWrite(i).req
    }
  } else {
    dmaBankWrite <> loaderBankWrite
    assert(!frontend.io.is_mvin_mmio_active, "MemDomain MMIO is disabled, but mvin_mmio was issued")

    for (i <- 0 until totalMmioRead) {
      io.ballDomain.mmioRead(i).req.ready  := false.B
      io.ballDomain.mmioRead(i).resp.valid := false.B
      io.ballDomain.mmioRead(i).resp.bits  := 0.U.asTypeOf(io.ballDomain.mmioRead(i).resp.bits)
    }
    for (i <- 0 until totalMmioWrite) {
      io.ballDomain.mmioWrite(i).req.ready := false.B
    }
  }

  // Shared path passthrough
  io.shared_mem_req <> backend.io.shared_mem_req
  io.shared_config <> backend.io.shared_config
}
