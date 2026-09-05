package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd.rs.{MemRsComplete, MemRsIssue}
import freechips.rocketchip.rocket.MStatus
import framework.memdomain.frontend.mem.dma.{BBWriteRequest, BBWriteResponse}
import framework.balldomain.blink.BankRead
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MemStorer(val b: GlobalConfig) extends Module {
  val rob_id_width = log2Up(b.frontend.rob_entries)

  // One bank line bytes
  private val line_bytes  = b.memDomain.bankWidth / 8
  // We pack/send 16B aligned beats to DMA
  private val align_bytes = 16
  require(isPow2(line_bytes), "MemStorer line_bytes must be power of 2")
  private val lgLine      = log2Ceil(line_bytes)

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val dmaReq  = Decoupled(new BBWriteRequest(b.memDomain.bankWidth))
    val dmaResp = Flipped(Decoupled(new BBWriteResponse))

    val bankRead = Flipped(new BankRead(b))

    val query_valid       = Output(Bool())
    val query_vbank_id    = Output(UInt(8.W))
    val query_is_shared   = Output(Bool())
    val query_group_count = Input(UInt(log2Up(b.memDomain.bankNum + 1).W))

    // Propagate decoded shared/private access intent.
    val is_shared = Output(Bool())
  })

  // -----------------------------
  // State
  // -----------------------------
  val s_idle :: s_setup :: s_query :: s_query_wait :: s_mul :: s_issue_sram_req :: s_wait_sram_resp :: s_have_sram_beat :: s_push_dma :: s_wait_dma_resp :: s_done :: Nil =
    Enum(11)
  val state                                                                                                                                                               = RegInit(s_idle)

  val rob_id_reg      = RegInit(0.U(rob_id_width.W))
  val is_sub_reg      = RegInit(false.B)
  val sub_rob_id_reg  = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val iter_reg        = RegInit(0.U(b.frontend.iter_len.W))
  val rd_bank_reg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val stride_reg      = RegInit(1.U(19.W))
  val group_count_reg = RegInit(1.U(log2Up(b.memDomain.bankNum + 1).W))
  val is_shared_reg   = RegInit(false.B)

  val addr_counter  = RegInit(0.U(b.frontend.iter_len.W))
  val group_counter = RegInit(0.U(log2Up(b.memDomain.bankNum + 1).W))
  val cursor        = RegInit(0.U(b.memDomain.memAddrLen.W))
  val rowDelta      = RegInit(0.U(b.memDomain.memAddrLen.W))

  // -----------------------------
  // Pending buffer for SRAM resp
  // -----------------------------
  val pending    = RegInit(false.B)
  val pendData   = Reg(UInt(b.memDomain.bankWidth.W))
  val pendIsLast = RegInit(false.B)

  val split_valid = RegInit(false.B)
  val split_addr  = RegInit(0.U(b.memDomain.memAddrLen.W))
  val split_data  = RegInit(0.U((align_bytes * 8).W))
  val split_mask  = RegInit(0.U(align_bytes.W))

  // Convenience
  val target_bank = rd_bank_reg

  // -----------------------------
  // Cmd accept
  // -----------------------------
  io.cmdReq.ready := (state === s_idle)

  when(io.cmdReq.fire && io.cmdReq.bits.cmd.is_store) {
    val strideIn = io.cmdReq.bits.cmd.special(57, 39)
    assert(strideIn >= 1.U, "MemStorer stride must be >= 1")
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
    rd_bank_reg    := io.cmdReq.bits.cmd.bank_id
    stride_reg     := strideIn
    iter_reg       := io.cmdReq.bits.cmd.iter
    is_shared_reg  := io.cmdReq.bits.cmd.is_shared

    addr_counter  := 0.U
    group_counter := 0.U
    cursor        := io.cmdReq.bits.cmd.mem_addr

    pending     := false.B
    split_valid := false.B
    split_addr  := 0.U
    split_data  := 0.U
    split_mask  := 0.U

    state := s_setup
  }

  io.query_valid     := state === s_setup
  io.query_vbank_id  := rd_bank_reg
  io.query_is_shared := is_shared_reg && (state === s_setup)

  when(state === s_setup) {
    state := s_query
  }

  when(state === s_query) {
    state := s_query_wait
  }

  when(state === s_query_wait) {
    assert(io.query_group_count >= 1.U, "MemStorer groups must be >= 1")
    group_count_reg := io.query_group_count
    state           := s_mul
  }

  when(state === s_mul) {
    rowDelta := ((group_count_reg * (stride_reg - 1.U)) + 1.U) << lgLine
    state    := s_issue_sram_req
  }

  // -----------------------------
  // SRAM read request
  // -----------------------------
  io.bankRead.rob_id   := rob_id_reg
  io.bankRead.bank_id  := target_bank
  io.bankRead.ball_id  := 0.U
  io.bankRead.group_id := group_counter(log2Up(b.memDomain.bankNum) - 1, 0)
  io.is_shared         := is_shared_reg

  io.bankRead.io.req.valid     := (state === s_issue_sram_req)
  io.bankRead.io.req.bits.addr := addr_counter

  // SRAMBank read resp is a 1-cycle pulse, so we must ALWAYS be ready to take it,
  // but only if we don't already hold a pending beat.
  io.bankRead.io.resp.ready := !pending

  when(state === s_issue_sram_req) {
    // Once request handshakes, wait for resp
    when(io.bankRead.io.req.fire) {
      state := s_wait_sram_resp
    }
  }

  // -----------------------------
  // Latch SRAM resp into pending (never drop it)
  // -----------------------------
  val bank_resp_fire = io.bankRead.io.resp.fire
  when(bank_resp_fire) {
    pending  := true.B
    pendData := io.bankRead.io.resp.bits.data
    // Last beat: last row and last group
    val is_last_row   = addr_counter >= iter_reg - 1.U
    val is_last_group = group_counter >= group_count_reg - 1.U
    pendIsLast := is_last_row && is_last_group && (iter_reg =/= 0.U)
    state      := s_have_sram_beat
  }

  val addr_offset = cursor(log2Ceil(align_bytes) - 1, 0)

  val aligned_addr = Cat(
    cursor(b.memDomain.memAddrLen - 1, log2Ceil(align_bytes)),
    0.U(log2Ceil(align_bytes).W)
  )

  val incoming_data = pendData
  val first_bytes   = align_bytes.U - addr_offset
  val first_mask    = (~0.U(align_bytes.W)) << addr_offset
  val second_mask   = (1.U(align_bytes.W) << addr_offset) - 1.U

  // -----------------------------
  // DMA request (Decoupled correct): hold valid until fire
  // -----------------------------
  val dma_v    = RegInit(false.B)
  val dma_addr = RegInit(0.U(b.memDomain.memAddrLen.W))
  val dma_data = RegInit(0.U((align_bytes * 8).W))
  val dma_mask = RegInit(0.U(align_bytes.W))

  io.dmaReq.valid       := dma_v
  io.dmaReq.bits.vaddr  := dma_addr
  io.dmaReq.bits.data   := dma_data
  io.dmaReq.bits.len    := align_bytes.U
  io.dmaReq.bits.mask   := dma_mask
  io.dmaReq.bits.status := 0.U.asTypeOf(new MStatus)

  io.dmaResp.ready := state === s_wait_dma_resp

  // When we have a pending SRAM beat, prepare one DMA beat (and keep it until fire)
  when(state === s_have_sram_beat) {
    // Only arm dma_v if not already armed
    when(!dma_v) {
      dma_v       := true.B
      dma_addr    := aligned_addr
      dma_data    := Mux(addr_offset === 0.U, incoming_data, incoming_data << (addr_offset * 8.U))
      dma_mask    := Mux(addr_offset === 0.U, ~0.U(align_bytes.W), first_mask)
      split_valid := addr_offset =/= 0.U
      split_addr  := aligned_addr + align_bytes.U
      split_data  := incoming_data >> (first_bytes * 8.U)
      split_mask  := second_mask
      state       := s_push_dma
    }
  }

  // When DMA accepts the beat, consume pending and move forward
  when(state === s_push_dma) {
    when(io.dmaReq.fire) {
      dma_v := false.B
      state := s_wait_dma_resp
    }
  }

  when(state === s_wait_dma_resp && io.dmaResp.fire) {
    when(split_valid) {
      dma_v       := true.B
      dma_addr    := split_addr
      dma_data    := split_data
      dma_mask    := split_mask
      split_valid := false.B
      state       := s_push_dma
    }.otherwise {
      pending := false.B

      val is_last_row   = addr_counter >= iter_reg - 1.U
      val is_last_group = group_counter >= group_count_reg - 1.U
      val all_done      = is_last_row && is_last_group && (iter_reg =/= 0.U)

      when(iter_reg =/= 0.U) {
        when(group_counter + 1.U < group_count_reg) {
          group_counter := group_counter + 1.U
          cursor        := cursor + line_bytes.U
        }.otherwise {
          group_counter := 0.U
          addr_counter  := addr_counter + 1.U
          cursor        := cursor + rowDelta
        }
      }

      when(pendIsLast || iter_reg === 0.U || all_done) {
        state := s_done
      }.otherwise {
        state := s_issue_sram_req
      }
    }
  }

  // If we are waiting for SRAM resp (but resp will pulse), just stay here
  when(state === s_wait_sram_resp) {
    // nothing; latch happens in bank_resp_fire block above
  }

  // -----------------------------
  // Completion
  // -----------------------------
  io.cmdResp.valid           := (state === s_done)
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  when(io.cmdResp.fire) {
    state := s_idle
  }
}
