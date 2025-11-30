package framework.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.memdomain.rs.{MemRsIssue, MemRsComplete}
import freechips.rocketchip.rocket.MStatus
import framework.memdomain.mem.SramReadIO
import framework.memdomain.dma.{BBWriteRequest, BBWriteResponse, LocalAddr}

class MemStorer(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  // Byte count of one row of data
  val line_bytes = b.spad_w / 8
  // 16-byte alignment
  val align_bytes = 16

  val io = IO(new Bundle {
    // Store instruction from ReservationStation
    val cmdReq = Flipped(Decoupled(new MemRsIssue))
    // Completion signal sent to ReservationStation
    val cmdResp = Decoupled(new MemRsComplete)
    // Direct connection to DMA write interface
    val dmaReq = Decoupled(new BBWriteRequest(b.spad_w))
    val dmaResp = Flipped(Decoupled(new BBWriteResponse))
    // Connected to Scratchpad SRAM read interface
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  })

  val s_idle :: s_sram_req :: s_dma_wait :: Nil = Enum(3)
  val state = RegInit(s_idle)

  val rob_id_reg = RegInit(0.U(rob_id_width.W))
  val mem_addr_reg = Reg(UInt(b.memAddrLen.W))
  val iter_reg = Reg(UInt(10.W))
  val sram_count = Reg(UInt(10.W))
  // Whether this is an acc bank operation
  val acc_reg = RegInit(false.B)
  // Used to alternately read two acc banks
  val acc_flip_reg = RegInit(false.B)
  // Cache stride
  val stride_reg = Reg(UInt(10.W))
  // Cache decoded bank information
  // Need 3 bits to support 8 banks (SPAD+ACC)
  val rd_bank_reg = Reg(UInt(log2Up(b.sp_banks + b.acc_banks).W))
  val rd_bank_addr_reg = Reg(UInt(log2Up(b.spad_bank_entries).W))

  // Data buffer related registers
  // 16-byte buffer
  val data_buffer = Reg(UInt((align_bytes * 8).W))
  // Number of valid bytes in buffer
  val buffer_valid_bytes = Reg(UInt(log2Ceil(align_bytes + 1).W))
  // Starting address corresponding to buffer
  val buffer_start_addr = Reg(UInt(b.memAddrLen.W))

  // Receive store instruction
  io.cmdReq.ready := state === s_idle

  when (io.cmdReq.fire && io.cmdReq.bits.cmd.is_store) {
    state         := s_sram_req
    rob_id_reg    := io.cmdReq.bits.rob_id
    mem_addr_reg  := io.cmdReq.bits.cmd.mem_addr
    iter_reg      := io.cmdReq.bits.cmd.iter
    rd_bank_reg   := io.cmdReq.bits.cmd.sp_bank
    rd_bank_addr_reg := io.cmdReq.bits.cmd.sp_bank_addr
    sram_count    := 0.U
    // Determine if acc based on bank
    acc_reg       := (io.cmdReq.bits.cmd.sp_bank >= b.sp_banks.U)
    // Reset flip register
    acc_flip_reg  := true.B
    stride_reg    := io.cmdReq.bits.cmd.special(10,0)
    // Initialize buffer state
    buffer_valid_bytes := 0.U
  }

  // Stream read SRAM data
  // Calculate current read bank and address
  val current_bank_addr = rd_bank_addr_reg + sram_count
  // All reads come from the same bank
  val target_bank = rd_bank_reg
  val target_row = current_bank_addr

  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid := (state === s_sram_req) && (target_bank === i.U) && !acc_reg
    io.sramRead(i).req.bits.addr := target_row
    io.sramRead(i).req.bits.fromDMA := true.B
  }

  // Default assignment
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits.addr := 0.U
    io.accRead(i).req.bits.fromDMA := true.B
  }

  for (i <- 0 until b.acc_banks/2){
    when((state === s_sram_req) && acc_reg){
      when(sram_count(2) === 0.U){
        io.accRead(i).req.valid := i.U === target_row(log2Ceil(b.acc_banks/2) - 1, 0)
        io.accRead(i).req.bits.addr := rd_bank_addr_reg + (sram_count >> (log2Ceil(b.acc_banks/2) + 1))
        io.accRead(i).req.bits.fromDMA := true.B
      }.otherwise{
        io.accRead(i + b.acc_banks/2).req.valid := i.U === target_row(log2Ceil(b.acc_banks/2) - 1, 0)
        io.accRead(i + b.acc_banks/2).req.bits.addr := rd_bank_addr_reg + (sram_count >> (log2Ceil(b.acc_banks/2) + 1))
        io.accRead(i + b.acc_banks/2).req.bits.fromDMA := true.B
      }
    }
  }

  // SRAM response processing
  val sram_resp_valid = io.sramRead.map(_.resp.valid).reduce(_ || _)
  val sram_resp_data = Mux1H(io.sramRead.map(_.resp.valid), io.sramRead.map(_.resp.bits.data))
  val acc_resp_valid = io.accRead.map(_.resp.valid).reduce(_ || _)
  val acc_resp_data = Mux1H(io.accRead.map(_.resp.valid), io.accRead.map(_.resp.bits.data))

  // Calculate memory address corresponding to current row
  val current_mem_addr = mem_addr_reg + sram_count(1,0) * line_bytes.U + ((sram_count >> 2) << 2) * stride_reg * line_bytes.U
  // Lower 4 bits of address, 0 when 16-byte aligned
  val addr_offset = current_mem_addr(log2Ceil(align_bytes) - 1, 0)
  val aligned_addr = Cat(current_mem_addr(b.memAddrLen - 1, log2Ceil(align_bytes)), 0.U(log2Ceil(align_bytes).W))
  val is_aligned = addr_offset === 0.U
  dontTouch(is_aligned)
  dontTouch(aligned_addr)

  // Data merge logic (line_bytes = 16 bytes)
  val incoming_data = Mux(sram_resp_valid, sram_resp_data.asUInt, acc_resp_data.asUInt)
  // Always 16 bytes
  val incoming_bytes = 16.U

  // Data merged into buffer
  val merged_data = Wire(UInt((align_bytes * 8).W))
  val total_valid_bytes = Wire(UInt(log2Ceil(align_bytes * 2).W))
  val is_last_iter = (sram_count >= (iter_reg - 1.U) && iter_reg > 0.U) || iter_reg === 0.U

  when (buffer_valid_bytes === 0.U) {
    // Buffer is empty
    when (addr_offset === 0.U) {
      // Address is aligned, use data directly
      merged_data := incoming_data
      total_valid_bytes := incoming_bytes
    }.otherwise {
      // Address not aligned, first time: use low bits of new data as high bits of send data, pad low bits with 0
      val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
      merged_data := new_data_low << (addr_offset * 8.U)
      total_valid_bytes := align_bytes.U
    }
  }.otherwise {
    // Buffer has data, concatenate: low bits of new data as high bits + buffer data as low bits
    val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
    merged_data := (new_data_low << (addr_offset * 8.U)) | data_buffer
    // Always 16 bytes
    total_valid_bytes := align_bytes.U
  }

  // Send logic: except for last iteration, can always fill 16 bytes
  val can_send_full_line = total_valid_bytes >= align_bytes.U
  val send_bytes = Mux(can_send_full_line, align_bytes.U, total_valid_bytes)

  // Determine send address - always use aligned address
  val send_addr = Mux(buffer_valid_bytes === 0.U, aligned_addr,
    Cat(buffer_start_addr(b.memAddrLen - 1, log2Ceil(align_bytes)), 0.U(log2Ceil(align_bytes).W)))

  // DMA request logic
  val should_send_normal = (sram_resp_valid || acc_resp_valid) && can_send_full_line
  val should_send_first_unaligned = (sram_resp_valid || acc_resp_valid) && (buffer_valid_bytes === 0.U && addr_offset =/= 0.U)
  val should_send_last = (sram_resp_valid || acc_resp_valid) && is_last_iter && !can_send_full_line
  val should_send = should_send_normal || should_send_first_unaligned || should_send_last

  // Add a flag to track whether all data has been processed
  // Completion detection logic - supports two cases:
  // 1. Clean data completion (fully aligned case)
  val aligned_completion = buffer_valid_bytes === 0.U && is_last_iter
  // 2. Remaining data needs to be sent (unaligned case)
  // Has remaining data
  val has_remaining_data_completion = buffer_valid_bytes > 0.U && is_last_iter
  val unaligned_completion_final_send = has_remaining_data_completion && !(sram_resp_valid || acc_resp_valid)

  // Generate mask
  val send_mask = Wire(UInt(align_bytes.W))
  when (buffer_valid_bytes === 0.U && addr_offset =/= 0.U) {
    // First unaligned: send high bits of new data, mask on high bits
    val valid_bytes = align_bytes.U - addr_offset
    // 0xFF00 (if addr_offset=8)
    send_mask := ((1.U << valid_bytes) - 1.U) << addr_offset
  }.elsewhen (buffer_valid_bytes > 0.U && can_send_full_line) {
    // Middle concatenation: send full 16 bytes
    send_mask := ~0.U(align_bytes.W)  // 0xFFFF
  }.elsewhen (unaligned_completion_final_send) {
    // Last send remaining buffer data: buffer data in low bits
    send_mask := (1.U << buffer_valid_bytes) - 1.U  // 0x00FF
  }.otherwise {
    // Aligned case: full data
    send_mask := ~0.U(align_bytes.W)  // 0xFFFF
  }

  // DMA request signal control logic - can only update when DMA is ready
  val dma_req_valid_reg = RegInit(false.B)
  val dma_req_vaddr_reg = RegInit(0.U(b.memAddrLen.W))
  val dma_req_data_reg = RegInit(0.U((align_bytes * 8).W))
  val dma_req_len_reg = RegInit(0.U(8.W))
  val dma_req_mask_reg = RegInit(0.U(align_bytes.W))
  val dma_req_status_reg = RegInit(0.U.asTypeOf(new MStatus))

  // Calculate DMA request signals
  val dma_req_valid_next = (should_send || unaligned_completion_final_send) && (state === s_sram_req || state === s_dma_wait)
  val dma_req_vaddr_next = Mux(unaligned_completion_final_send, buffer_start_addr, send_addr)
  val dma_req_data_next = Mux(unaligned_completion_final_send, data_buffer, merged_data)
  val dma_req_len_next = align_bytes.U
  val dma_req_mask_next = Mux(unaligned_completion_final_send, (1.U << buffer_valid_bytes) - 1.U, send_mask)
  val dma_req_status_next = 0.U.asTypeOf(new MStatus)

  // Only update registers when DMA is ready
  when (io.dmaReq.ready) {
    dma_req_valid_reg := dma_req_valid_next
    dma_req_vaddr_reg := dma_req_vaddr_next
    dma_req_data_reg := dma_req_data_next
    dma_req_len_reg := dma_req_len_next
    dma_req_mask_reg := dma_req_mask_next
    dma_req_status_reg := dma_req_status_next
  }

  // Connect to DMA interface
  io.dmaReq.valid := dma_req_valid_reg
  io.dmaReq.bits.vaddr := dma_req_vaddr_reg
  io.dmaReq.bits.data := dma_req_data_reg
  io.dmaReq.bits.len := dma_req_len_reg
  io.dmaReq.bits.mask := dma_req_mask_reg
  io.dmaReq.bits.status := dma_req_status_reg

  // Connect SRAM response ready signal - based on DMA ready state
  io.sramRead.foreach(_.resp.ready := io.dmaReq.ready && (state === s_sram_req || state === s_dma_wait))
  io.accRead.foreach(_.resp.ready := io.dmaReq.ready && (state === s_sram_req || state === s_dma_wait))
  // State transition and counter update
  when (io.sramRead.map(_.req.fire).reduce(_ || _)) {
    state := s_dma_wait
  }

  when (io.dmaReq.fire) {
    when (!unaligned_completion_final_send) {
      sram_count := sram_count + 1.U
    }

    // Update buffer state
    when (addr_offset =/= 0.U && (sram_resp_valid || acc_resp_valid)) {
      // Unaligned case: cache high bits of new data
      // Cache is the high bits part
      val remaining_bytes = align_bytes.U - addr_offset
      data_buffer := incoming_data >> (addr_offset * 8.U)
      buffer_valid_bytes := remaining_bytes
      // Update buffer corresponding address (point to next 16-byte aligned address)
      when (buffer_valid_bytes === 0.U) {
        buffer_start_addr := aligned_addr + align_bytes.U
      }.otherwise {
        buffer_start_addr := buffer_start_addr + align_bytes.U
      }
    }.elsewhen (unaligned_completion_final_send) {
      // Sent final remaining data, clear buffer
      buffer_valid_bytes := 0.U
    }.otherwise {
      // In aligned case, if previous buffer data was merged and sent, need to clear buffer
      when (buffer_valid_bytes > 0.U && can_send_full_line && sram_resp_valid) {
        buffer_valid_bytes := 0.U
      }
    }

    // Fix state transition logic
    when (unaligned_completion_final_send) {
      // Only return to idle after unaligned_completion_final_send completes
      state := s_idle
    }.elsewhen (aligned_completion) {
      // All data has been sent
      state := s_idle
    }.elsewhen (sram_count + 1.U >= iter_reg && iter_reg > 0.U) {
      // Iteration ended, but there may still be buffer data to send
      when (buffer_valid_bytes > 0.U) {
        // Maintain state, wait for unaligned_completion_final_send
        state := s_dma_wait
      }.otherwise {
        state := s_idle
      }
    }.elsewhen (iter_reg === 0.U) {
      state := s_idle
    }.otherwise {
      state := s_sram_req
    }
  }

  // Wait for DMA to truly complete
  io.dmaResp.ready := true.B

  // Fix completion signal logic - only issue completion signal after all data transfer is truly complete
  val task_complete = RegInit(false.B)
  when (io.cmdReq.fire && io.cmdReq.bits.cmd.is_store) {
    task_complete := false.B
  }.elsewhen (io.dmaReq.fire && (unaligned_completion_final_send || aligned_completion)) {
    task_complete := true.B
  }

  io.cmdResp.valid := task_complete && (state === s_idle)
  io.cmdResp.bits.rob_id := rob_id_reg

  // Reset flag after sending completion signal
  when (io.cmdResp.fire) {
    task_complete := false.B
  }
}
