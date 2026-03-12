package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import freechips.rocketchip.rocket.MStatus
import framework.memdomain.frontend.outside_channel.dma.{BBWriteRequest, BBWriteResponse}
import framework.balldomain.blink.BankRead
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MemStorer(val b: GlobalConfig) extends Module {
  val rob_id_width = log2Up(b.frontend.rob_entries)

  // One bank line bytes
  private val line_bytes  = b.memDomain.bankWidth / 8
  // We pack/send 16B aligned beats to DMA
  private val align_bytes = 16

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val dmaReq  = Decoupled(new BBWriteRequest(b.memDomain.bankWidth))
    val dmaResp = Flipped(Decoupled(new BBWriteResponse))

    val bankRead = Flipped(new BankRead(b))

    // Query interface to get group count
    val query_vbank_id    = Output(UInt(8.W))
    val query_is_shared   = Output(Bool())
    val query_group_count = Input(UInt(4.W))

    // Propagate decoded shared/private access intent.
    val is_shared = Output(Bool())
  })

  // -----------------------------
  // State
  // -----------------------------
  val s_idle :: s_issue_sram_req :: s_wait_sram_resp :: s_have_sram_beat :: s_push_dma :: s_done :: Nil = Enum(6)
  val state                                                                                             = RegInit(s_idle)

  val rob_id_reg      = RegInit(0.U(rob_id_width.W))
  val mem_addr_reg    = RegInit(0.U(b.memDomain.memAddrLen.W))
  val iter_reg        = RegInit(0.U(b.frontend.iter_len.W))
  val stride_reg      = RegInit(0.U(10.W))
  val rd_bank_reg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val group_count_reg = RegInit(1.U(4.W)) // Store group count for current operation
  val is_shared_reg   = RegInit(false.B)

  // Address and group counters
  val addr_counter  = RegInit(0.U(b.frontend.iter_len.W)) // Row address counter
  val group_counter = RegInit(0.U(4.W))                   // Group counter within a row

  // -----------------------------
  // Pending buffer for SRAM resp
  // -----------------------------
  val pending    = RegInit(false.B)
  val pendData   = Reg(UInt(b.memDomain.bankWidth.W))
  val pendIsLast = RegInit(false.B)

  // -----------------------------
  // Optional: simple 16B align/merge support (keep your original intent)
  // We'll keep a small byte buffer for unaligned head/tail.
  // -----------------------------
  val data_buffer        = RegInit(0.U((align_bytes * 8).W)) // 16B
  val buffer_valid_bytes = RegInit(0.U(log2Ceil(align_bytes + 1).W))
  val buffer_start_addr  = RegInit(0.U(b.memDomain.memAddrLen.W))

  // Convenience
  val target_bank = rd_bank_reg

  // -----------------------------
  // Cmd accept
  // -----------------------------
  io.cmdReq.ready := (state === s_idle)

  when(io.cmdReq.fire && io.cmdReq.bits.cmd.is_store) {
    rob_id_reg   := io.cmdReq.bits.rob_id
    mem_addr_reg := io.cmdReq.bits.cmd.mem_addr
    rd_bank_reg  := io.cmdReq.bits.cmd.bank_id
    stride_reg   := io.cmdReq.bits.cmd.special(57, 39)

    // Query and save group count
    group_count_reg := io.query_group_count
    iter_reg        := io.cmdReq.bits.cmd.iter
    is_shared_reg   := io.cmdReq.bits.cmd.is_shared

    // Initialize counters
    addr_counter  := 0.U
    group_counter := 0.U

    pending            := false.B
    data_buffer        := 0.U
    buffer_valid_bytes := 0.U
    buffer_start_addr  := 0.U

    state := s_issue_sram_req
  }

  // Drive query interface
  // When idle and cmdReq is valid, query the incoming bank_id
  // Otherwise use the registered bank_id
  io.query_vbank_id  := Mux(state === s_idle && io.cmdReq.valid, io.cmdReq.bits.cmd.bank_id, rd_bank_reg)
  io.query_is_shared := Mux(state === s_idle && io.cmdReq.valid, io.cmdReq.bits.cmd.is_shared, is_shared_reg)

  // -----------------------------
  // SRAM read request
  // -----------------------------
  io.bankRead.rob_id   := rob_id_reg
  io.bankRead.bank_id  := target_bank
  io.bankRead.ball_id  := 0.U
  io.bankRead.group_id := group_counter
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

  // -----------------------------
  // Address calculation
  // For multi-bank: each row has group_count groups, each group is line_bytes
  // So: offset = addr * (group_count * line_bytes) + group * line_bytes
  // -----------------------------
  val current_mem_addr =
    mem_addr_reg +
      addr_counter * group_count_reg * line_bytes.U +
      group_counter * line_bytes.U

  val addr_offset = current_mem_addr(log2Ceil(align_bytes) - 1, 0)

  val aligned_addr = Cat(
    current_mem_addr(b.memDomain.memAddrLen - 1, log2Ceil(align_bytes)),
    0.U(log2Ceil(align_bytes).W)
  )

  // -----------------------------
  // Merge logic (kept compatible with your original behavior)
  // incoming_data is always 16 bytes (bankWidth==128 in your waveforms)
  // -----------------------------
  val incoming_data  = pendData
  val incoming_bytes = align_bytes.U

  val merged_data       = Wire(UInt((align_bytes * 8).W))
  val total_valid_bytes = Wire(UInt(log2Ceil(align_bytes * 2).W))

  when(buffer_valid_bytes === 0.U) {
    when(addr_offset === 0.U) {
      merged_data       := incoming_data
      total_valid_bytes := incoming_bytes
    }.otherwise {
      // first unaligned: send high part, pad low with 0
      val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
      merged_data       := new_data_low << (addr_offset * 8.U)
      total_valid_bytes := align_bytes.U
    }
  }.otherwise {
    val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
    merged_data       := (new_data_low << (addr_offset * 8.U)) | data_buffer
    total_valid_bytes := align_bytes.U
  }

  val can_send_full_line = total_valid_bytes >= align_bytes.U

  // send address (aligned)
  val send_addr = Mux(
    buffer_valid_bytes === 0.U,
    aligned_addr,
    Cat(buffer_start_addr(b.memDomain.memAddrLen - 1, log2Ceil(align_bytes)), 0.U(log2Ceil(align_bytes).W))
  )

  // send mask
  val send_mask = Wire(UInt(align_bytes.W))
  when(buffer_valid_bytes === 0.U && addr_offset =/= 0.U) {
    val valid_bytes = align_bytes.U - addr_offset
    send_mask := ((1.U << valid_bytes) - 1.U) << addr_offset
  }.elsewhen(buffer_valid_bytes > 0.U && can_send_full_line) {
    send_mask := ~0.U(align_bytes.W)
  }.otherwise {
    send_mask := ~0.U(align_bytes.W)
  }

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

  // By default we don't care dmaResp in this simple model
  io.dmaResp.ready := true.B

  // When we have a pending SRAM beat, prepare one DMA beat (and keep it until fire)
  when(state === s_have_sram_beat) {
    // Only arm dma_v if not already armed
    when(!dma_v) {
      dma_v    := true.B
      dma_addr := send_addr
      dma_data := merged_data
      dma_mask := send_mask
      state    := s_push_dma
    }
  }

  // When DMA accepts the beat, consume pending and move forward
  when(state === s_push_dma) {
    when(io.dmaReq.fire) {
      dma_v := false.B

      // Update buffer state like your original:
      when(addr_offset =/= 0.U) {
        val remaining_bytes = align_bytes.U - addr_offset
        data_buffer        := incoming_data >> (addr_offset * 8.U)
        buffer_valid_bytes := remaining_bytes
        when(buffer_valid_bytes === 0.U) {
          buffer_start_addr := aligned_addr + align_bytes.U
        }.otherwise {
          buffer_start_addr := buffer_start_addr + align_bytes.U
        }
      }.otherwise {
        // aligned: clear buffer if it was used
        when(buffer_valid_bytes > 0.U && can_send_full_line) {
          buffer_valid_bytes := 0.U
          data_buffer        := 0.U
        }
      }

      // Mark current beat consumed
      pending := false.B

      // Check if this was the last beat before advancing counters
      val is_last_row   = addr_counter >= iter_reg - 1.U
      val is_last_group = group_counter >= group_count_reg - 1.U
      val all_done      = is_last_row && is_last_group && (iter_reg =/= 0.U)

      // Advance counters
      when(iter_reg =/= 0.U) {
        when(group_counter + 1.U < group_count_reg) {
          // Move to next group in same row
          group_counter := group_counter + 1.U
        }.otherwise {
          // Move to next row, reset group counter
          group_counter := 0.U
          addr_counter  := addr_counter + 1.U
        }
      }

      // Decide next state based on completion check done BEFORE counter update
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
  io.cmdResp.valid       := (state === s_done)
  io.cmdResp.bits.rob_id := rob_id_reg

  when(io.cmdResp.fire) {
    state := s_idle
  }
}
