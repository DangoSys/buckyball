package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import framework.memdomain.frontend.cmd.rs.{MemRsComplete, MemRsIssue}
import framework.memdomain.backend.banks.SramWriteIO
import framework.memdomain.frontend.mem.dma.{BBReadRequest, BBReadResponse}
import freechips.rocketchip.rocket.MStatus
import framework.balldomain.blink.BankWrite
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class MemLoader(val b: GlobalConfig) extends Module {
  val rob_id_width = log2Up(b.frontend.rob_entries)

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val dmaReq  = Decoupled(new BBReadRequest())
    val dmaResp = Flipped(Decoupled(new BBReadResponse(b.memDomain.bankWidth)))

    val bankWrite = Flipped(new BankWrite(b))

    // MMIO routing hint: tells upper level (MemDomain) to route bankWrite to MmioPool
    // When is_mvin_mmio_active is true, mmio_addr/col carry MMIO destination info
    val is_mvin_mmio_active = Output(Bool())
    val mmio_addr           = Output(UInt(17.W)) // rs2[55:39]: MMIO byte address
    val mmio_col            = Output(UInt(8.W))  // rs2[63:56]: valid bytes per row

    val query_valid       = Output(Bool())
    val query_vbank_id    = Output(UInt(8.W))
    val query_is_shared   = Output(Bool())
    val query_group_count = Input(UInt(log2Up(b.memDomain.bankNum + 1).W))

    // Propagate decoded shared/private access intent.
    val is_shared = Output(Bool())
  })

  val s_idle :: s_setup :: s_mul :: s_dma_req :: s_dma_wait :: s_wait_write_resp :: s_done :: Nil = Enum(7)
  val state                                                                                       = RegInit(s_idle)

  val rob_id_reg     = RegInit(0.U(rob_id_width.W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val mem_addr_reg   = Reg(UInt(b.memDomain.memAddrLen.W))
  val iter_cmd       = Reg(UInt(b.frontend.iter_len.W))
  val iter_reg       = Reg(UInt(b.frontend.iter_len.W))
  val resp_count     = RegInit(0.U(log2Up(16).W))
  val wr_bank_reg    = Reg(UInt(log2Up(b.memDomain.bankNum).W))
  val stride_reg     = Reg(UInt(19.W))
  val is_shared_reg  = RegInit(false.B)

  // Group counter for multi-bank writes
  val group_counter   = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val group_count_reg = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val rowAddr         = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))

  // MMIO routing info (latched at cmdReq.fire, exposed to upper level)
  val is_mvin_mmio_reg = RegInit(false.B)
  val mmio_addr_reg    = RegInit(0.U(17.W))
  val mmio_col_reg     = RegInit(0.U(8.W))

  io.is_mvin_mmio_active := is_mvin_mmio_reg
  io.mmio_addr           := mmio_addr_reg
  io.mmio_col            := mmio_col_reg

  // -----------------------------
  // pending latch for 1-beat DMA -> bankWrite
  // -----------------------------
  val pending = RegInit(false.B)
  val latData = Reg(UInt(b.memDomain.bankWidth.W))
  val latLast = RegInit(false.B)

  // -----------------------------
  // defaults
  // -----------------------------
  io.cmdReq.ready := (state === s_idle)

  io.dmaReq.valid       := (state === s_dma_req)
  io.dmaReq.bits.vaddr  := mem_addr_reg
  io.dmaReq.bits.len    := iter_reg * (b.memDomain.bankWidth / 8).U
  io.dmaReq.bits.status := 0.U.asTypeOf(new MStatus)
  io.dmaReq.bits.stride := stride_reg
  io.dmaReq.bits.groups := group_count_reg

  // only accept DMA beat when waiting AND no pending beat buffered
  io.dmaResp.ready := (state === s_dma_wait) && !pending

  // bank write request driven from pending
  io.bankWrite.io.req.valid     := pending
  io.bankWrite.io.req.bits.addr := rowAddr
  io.bankWrite.io.req.bits.data := latData
  io.bankWrite.io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))

  // IMPORTANT: always ready for write response (avoid deadlock)
  io.bankWrite.io.resp.ready := true.B

  io.bankWrite.rob_id   := rob_id_reg
  io.bankWrite.bank_id  := wr_bank_reg
  io.bankWrite.ball_id  := 0.U
  io.bankWrite.group_id := group_counter(log2Up(b.memDomain.bankNum) - 1, 0)
  io.is_shared          := is_shared_reg

  // cmdResp (Decoupled): hold valid until accepted
  io.cmdResp.valid           := (state === s_done)
  io.cmdResp.bits            := 0.U.asTypeOf(new MemRsComplete(b))
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  // -----------------------------
  // Receive load instruction (both mvin and mvin_mmio go through is_load path)
  // -----------------------------
  when(io.cmdReq.fire && io.cmdReq.bits.cmd.is_load) {
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
    mem_addr_reg   := io.cmdReq.bits.cmd.mem_addr
    wr_bank_reg    := io.cmdReq.bits.cmd.bank_id
    resp_count     := 0.U
    pending        := false.B
    latLast        := false.B
    group_counter  := 0.U
    rowAddr        := 0.U
    is_shared_reg  := io.cmdReq.bits.cmd.is_shared

    is_mvin_mmio_reg := io.cmdReq.bits.cmd.is_mvin_mmio
    when(io.cmdReq.bits.cmd.is_mvin_mmio) {
      mmio_addr_reg   := io.cmdReq.bits.cmd.special(55, 39)
      mmio_col_reg    := io.cmdReq.bits.cmd.special(63, 56)
      iter_reg        := io.cmdReq.bits.cmd.iter
      group_count_reg := 1.U
      stride_reg      := 1.U
      state           := s_dma_req
    }.otherwise {
      stride_reg := io.cmdReq.bits.cmd.special(57, 39)
      iter_cmd   := io.cmdReq.bits.cmd.iter
      state      := s_setup
    }
  }

  io.query_valid     := state === s_setup
  io.query_vbank_id  := wr_bank_reg
  io.query_is_shared := is_shared_reg && (state === s_setup)

  when(state === s_setup) {
    assert(io.query_group_count >= 1.U, "MemLoader groups must be >= 1")
    group_count_reg := io.query_group_count
    state           := s_mul
  }

  when(state === s_mul) {
    iter_reg := iter_cmd * group_count_reg
    state    := s_dma_req
  }

  // DMA req accepted
  when(io.dmaReq.fire) {
    state      := s_dma_wait
    resp_count := 0.U
  }

  // Latch DMA beat into pending buffer
  when(io.dmaResp.fire) {
    pending := true.B
    latData := io.dmaResp.bits.data
    latLast := io.dmaResp.bits.last
  }

  // When bankWrite request handshakes, consume pending beat
  when(io.bankWrite.io.req.fire) {
    pending    := false.B
    resp_count := resp_count + 1.U

    when(latLast) {
      state := s_wait_write_resp
    }.otherwise {
      state := s_wait_write_resp
    }
  }

  // Wait for each write response before accepting the next DMA beat.
  when(state === s_wait_write_resp && io.bankWrite.io.resp.fire) {
    when(group_counter + 1.U < group_count_reg) {
      group_counter := group_counter + 1.U
    }.otherwise {
      group_counter := 0.U
      rowAddr       := rowAddr + 1.U
    }
    state := Mux(latLast, s_done, s_dma_wait)
  }

  when(state === s_done && io.cmdResp.fire) {
    state := s_idle
  }
}
