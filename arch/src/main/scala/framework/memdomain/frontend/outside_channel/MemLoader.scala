package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import framework.memdomain.backend.banks.SramWriteIO
import framework.memdomain.frontend.outside_channel.dma.{BBReadRequest, BBReadResponse}
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

    // Query interface to get group count
    val query_vbank_id    = Output(UInt(8.W))
    val query_group_count = Input(UInt(4.W))

    // Propagate decoded shared/private access intent.
    val is_shared = Output(Bool())
  })

  val s_idle :: s_dma_req :: s_dma_wait :: s_wait_last_write :: s_done :: Nil = Enum(5)
  val state                                                                   = RegInit(s_idle)

  val rob_id_reg    = RegInit(0.U(rob_id_width.W))
  val mem_addr_reg  = Reg(UInt(b.memDomain.memAddrLen.W))
  val iter_reg      = Reg(UInt(b.frontend.iter_len.W))
  val resp_count    = RegInit(0.U(log2Up(16).W))
  val wr_bank_reg   = Reg(UInt(log2Up(b.memDomain.bankNum).W))
  val stride_reg    = Reg(UInt(11.W))
  val is_shared_reg = RegInit(false.B)

  // Group counter for multi-bank writes
  val group_counter   = RegInit(0.U(4.W))
  val group_count_reg = RegInit(0.U(4.W))

  // -----------------------------
  // pending latch for 1-beat DMA -> bankWrite
  // -----------------------------
  val pending = RegInit(false.B)
  val latRow  = Reg(UInt(log2Up(b.memDomain.bankEntries).W))
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

  // only accept DMA beat when waiting AND no pending beat buffered
  io.dmaResp.ready := (state === s_dma_wait) && !pending

  // bank write request driven from pending
  io.bankWrite.io.req.valid      := pending
  io.bankWrite.io.req.bits.addr  := latRow / group_count_reg
  io.bankWrite.io.req.bits.data  := latData
  io.bankWrite.io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite.io.req.bits.wmode := false.B

  // IMPORTANT: always ready for write response (avoid deadlock)
  io.bankWrite.io.resp.ready := true.B

  io.bankWrite.rob_id   := rob_id_reg
  io.bankWrite.bank_id  := wr_bank_reg
  io.bankWrite.ball_id  := 0.U
  io.bankWrite.group_id := group_counter
  io.is_shared          := is_shared_reg

  // cmdResp (Decoupled): hold valid until accepted
  io.cmdResp.valid       := (state === s_done)
  io.cmdResp.bits        := 0.U.asTypeOf(new MemRsComplete(b))
  io.cmdResp.bits.rob_id := rob_id_reg

  // -----------------------------
  // Receive load instruction
  // -----------------------------
  when(io.cmdReq.fire && io.cmdReq.bits.cmd.is_load) {
    state           := s_dma_req
    rob_id_reg      := io.cmdReq.bits.rob_id
    mem_addr_reg    := io.cmdReq.bits.cmd.mem_addr
    wr_bank_reg     := io.cmdReq.bits.cmd.bank_id
    // stride from rs2[57:39]
    stride_reg      := io.cmdReq.bits.cmd.special(57, 39)
    resp_count      := 0.U
    pending         := false.B
    latLast         := false.B
    group_counter   := 0.U
    group_count_reg := io.query_group_count
    is_shared_reg   := io.cmdReq.bits.cmd.is_shared

    // Query group count and multiply iter
    iter_reg := io.cmdReq.bits.cmd.iter * io.query_group_count
  }

  // Drive query interface
  // When idle and cmdReq is valid, query the incoming bank_id
  // Otherwise use the registered bank_id
  io.query_vbank_id := Mux(state === s_idle && io.cmdReq.valid, io.cmdReq.bits.cmd.bank_id, wr_bank_reg)

  // DMA req accepted
  when(io.dmaReq.fire) {
    state      := s_dma_wait
    resp_count := 0.U
  }

  // Latch DMA beat into pending buffer
  when(io.dmaResp.fire) {
    pending := true.B
    latRow  := io.dmaResp.bits.addrcounter
    latData := io.dmaResp.bits.data
    latLast := io.dmaResp.bits.last
  }

  // When bankWrite request handshakes, consume pending beat
  when(io.bankWrite.io.req.fire) {
    pending    := false.B
    resp_count := resp_count + 1.U

    // Update group_counter
    when(group_counter + 1.U < group_count_reg) {
      group_counter := group_counter + 1.U
    }.otherwise {
      group_counter := 0.U
    }

    when(latLast) {
      // Last beat request sent, now wait for write response
      state := s_wait_last_write
    }
  }

  // Wait for the last write response before completing
  when(state === s_wait_last_write && io.bankWrite.io.resp.fire) {
    state := s_done
  }

  when(state === s_done && io.cmdResp.fire) {
    state := s_idle
  }
}
