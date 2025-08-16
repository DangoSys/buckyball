package framework.builtin.store

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.memdomain.{MemReservationStationIssue, MemReservationStationComplete, MemBuckyBallCmd}
import framework.builtin.mem.{SimpleWriteRequest, SimpleWriteResponse, SramReadIO, LocalAddr}
import framework.builtin.frontend.FrontendTLBIO
import freechips.rocketchip.rocket.MStatus
import framework.builtin.mem.{SimpleReadRequest, SimpleReadResponse, SramReadIO}

class MemStorer(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  val line_bytes = b.spad_w / 8  // 一行数据的字节数
  val align_bytes = 16  // 16字节对齐
  
  val io = IO(new Bundle {
    // 来自ReservationStation的store指令
    val cmdReq = Flipped(Decoupled(new MemReservationStationIssue))
    // 发送给ReservationStation的完成信号
    val cmdResp = Decoupled(new MemReservationStationComplete)
    // 直接连接DMA写入接口
    val dmaReq = Decoupled(new SimpleWriteRequest(b.spad_w))
    val dmaResp = Flipped(Decoupled(new SimpleWriteResponse))
    // 连接到Scratchpad的SRAM读取接口
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  })

  val s_idle :: s_sram_req :: s_dma_wait :: Nil = Enum(3)
  val state = RegInit(s_idle)
  
  val rob_id_reg = RegInit(0.U(rob_id_width.W))
  val mem_addr_reg = Reg(UInt(b.memAddrLen.W))
  val iter_reg = Reg(UInt(10.W))
  val sram_count = Reg(UInt(10.W))
  val acc_reg = RegInit(false.B)  // 是否是acc bank的操作
  
  // 缓存解码好的bank信息
  val rd_bank_reg = Reg(UInt(log2Up(b.sp_banks).W))
  val rd_bank_addr_reg = Reg(UInt(log2Up(b.spad_bank_entries).W))

  // 数据缓存相关寄存器
  val data_buffer = Reg(UInt((align_bytes * 8).W))  // 16字节缓存
  val buffer_valid_bytes = Reg(UInt(log2Ceil(align_bytes + 1).W))  // 缓存中有效字节数
  val buffer_start_addr = Reg(UInt(b.memAddrLen.W))  // 缓存对应的起始地址
  
  // 接收store指令
  io.cmdReq.ready := state === s_idle
  
  when (io.cmdReq.fire && io.cmdReq.bits.cmd.mem_decode_cmd.is_store) {
    state := s_sram_req
    rob_id_reg := io.cmdReq.bits.rob_id
    mem_addr_reg := io.cmdReq.bits.cmd.mem_decode_cmd.mem_addr
    iter_reg := io.cmdReq.bits.cmd.mem_decode_cmd.iter
    rd_bank_reg := io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank
    rd_bank_addr_reg := io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank_addr
    sram_count := 0.U
    acc_reg := (io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank >= b.sp_banks.U) // 根据bank判断是否是acc
    
    // 初始化缓存状态
    buffer_valid_bytes := 0.U
  }

  // 流式读取SRAM数据
  // 计算当前读取的bank和地址
  val current_bank_addr = rd_bank_addr_reg + sram_count
  val target_bank = rd_bank_reg  // 所有读取都来自同一个bank
  val target_row = current_bank_addr
  
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid := (state === s_sram_req) && (target_bank === i.U) && !acc_reg
    io.sramRead(i).req.bits.addr := target_row
    io.sramRead(i).req.bits.fromDMA := true.B
  }

  for(i <- 0 until b.acc_banks){
    io.accRead(i).req.valid := (state === s_sram_req) && acc_reg && (i.U === target_row(log2Ceil(b.acc_banks) - 1, 0))
    io.accRead(i).req.bits.addr := rd_bank_addr_reg + (sram_count >> log2Ceil(b.acc_banks))
    io.accRead(i).req.bits.fromDMA := true.B
  }

  // SRAM响应处理
  val sram_resp_valid = io.sramRead.map(_.resp.valid).reduce(_ || _)
  val sram_resp_data = Mux1H(io.sramRead.map(_.resp.valid), io.sramRead.map(_.resp.bits.data))
  val acc_resp_valid = io.accRead.map(_.resp.valid).reduce(_ || _)
  val acc_resp_data = Mux1H(io.accRead.map(_.resp.valid), io.accRead.map(_.resp.bits.data))

  // 计算当前行对应的内存地址
  val current_mem_addr = mem_addr_reg + (sram_count * line_bytes.U)
  val addr_offset = current_mem_addr(log2Ceil(align_bytes) - 1, 0)  // 地址的低4位，16字节对齐时为0
  val aligned_addr = Cat(current_mem_addr(b.memAddrLen - 1, log2Ceil(align_bytes)), 0.U(log2Ceil(align_bytes).W))
  val is_aligned = addr_offset === 0.U
  dontTouch(is_aligned)
  dontTouch(aligned_addr)
  
  // 数据合并逻辑 (line_bytes = 16字节)
  val incoming_data = Mux(sram_resp_valid, sram_resp_data.asUInt, acc_resp_data.asUInt)
  val incoming_bytes = 16.U  // 永远是16字节
  
  // 合并到缓存的数据
  val merged_data = Wire(UInt((align_bytes * 8).W))
  val total_valid_bytes = Wire(UInt(log2Ceil(align_bytes * 2).W))
  val is_last_iter = (sram_count >= iter_reg && iter_reg > 0.U) || iter_reg === 0.U
  
  when (buffer_valid_bytes === 0.U) {
    // 缓存为空
    when (addr_offset === 0.U) {
      // 地址已对齐，直接使用数据
      merged_data := incoming_data
      total_valid_bytes := incoming_bytes
    }.otherwise {
      // 地址不对齐，第一次：将新数据低位作为发送数据高位，低位补0
      val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
      merged_data := new_data_low << (addr_offset * 8.U)
      total_valid_bytes := align_bytes.U
    }
  }.otherwise {
    // 缓存有数据，拼接：新数据低位作为高位 + 缓存数据作为低位
    val new_data_low = incoming_data & ((1.U << (addr_offset * 8.U)) - 1.U)
    merged_data := (new_data_low << (addr_offset * 8.U)) | data_buffer
    total_valid_bytes := align_bytes.U  // 总是16字节
  }
  
  // 发送逻辑：除了最后一次迭代，总是能填满16字节
  val can_send_full_line = total_valid_bytes >= align_bytes.U
  val send_bytes = Mux(can_send_full_line, align_bytes.U, total_valid_bytes)
  
  // 确定发送地址 - 始终使用对齐地址
  val send_addr = Mux(buffer_valid_bytes === 0.U, aligned_addr, 
    Cat(buffer_start_addr(b.memAddrLen - 1, log2Ceil(align_bytes)), 0.U(log2Ceil(align_bytes).W)))
  
  // DMA请求逻辑
  val should_send_normal = (sram_resp_valid || acc_resp_valid) && can_send_full_line
  val should_send_first_unaligned = (sram_resp_valid || acc_resp_valid) && (buffer_valid_bytes === 0.U && addr_offset =/= 0.U)
  val should_send_last = (sram_resp_valid || acc_resp_valid) && is_last_iter && !can_send_full_line
  val should_send = should_send_normal || should_send_first_unaligned || should_send_last
  
  // 迭代结束后还需要发送剩余缓存数据
  val has_remaining_data = buffer_valid_bytes > 0.U && is_last_iter
  val final_send = has_remaining_data && !(sram_resp_valid || acc_resp_valid)
  
  // 添加一个标志来跟踪是否所有数据都已处理完成
  val all_iterations_complete = (sram_count >= iter_reg && iter_reg > 0.U) || (iter_reg === 0.U)
  val all_data_sent = all_iterations_complete && buffer_valid_bytes === 0.U
  
  // 生成mask
  val send_mask = Wire(UInt(align_bytes.W))
  when (buffer_valid_bytes === 0.U && addr_offset =/= 0.U) {
    // 第一次非对齐：发送新数据高位，mask在高位
    val valid_bytes = align_bytes.U - addr_offset
    send_mask := ((1.U << valid_bytes) - 1.U) << addr_offset  // 0xFF00 (如果addr_offset=8)
  }.elsewhen (buffer_valid_bytes > 0.U && can_send_full_line) {
    // 中间拼接：发送完整16字节
    send_mask := ~0.U(align_bytes.W)  // 0xFFFF  
  }.elsewhen (final_send) {
    // 最后发送剩余buffer数据：缓存数据在低位
    send_mask := (1.U << buffer_valid_bytes) - 1.U  // 0x00FF
  }.otherwise {
    // 对齐情况：完整数据
    send_mask := ~0.U(align_bytes.W)  // 0xFFFF
  }
  
  // DMA请求信号控制逻辑 - 只有在DMA ready时才能更新
  val dma_req_valid_reg = RegInit(false.B)
  val dma_req_vaddr_reg = RegInit(0.U(b.memAddrLen.W))
  val dma_req_data_reg = RegInit(0.U((align_bytes * 8).W))
  val dma_req_len_reg = RegInit(0.U(8.W))
  val dma_req_mask_reg = RegInit(0.U(align_bytes.W))
  val dma_req_status_reg = RegInit(0.U.asTypeOf(new MStatus))
  
  // 计算DMA请求信号
  val dma_req_valid_next = (should_send || final_send) && (state === s_sram_req || state === s_dma_wait)
  val dma_req_vaddr_next = Mux(final_send, buffer_start_addr, send_addr)
  val dma_req_data_next = Mux(final_send, data_buffer, merged_data)
  val dma_req_len_next = align_bytes.U
  val dma_req_mask_next = Mux(final_send, (1.U << buffer_valid_bytes) - 1.U, send_mask)
  val dma_req_status_next = 0.U.asTypeOf(new MStatus)
  
  // 只有在DMA ready时才更新寄存器
  when (io.dmaReq.ready) {
    dma_req_valid_reg := dma_req_valid_next
    dma_req_vaddr_reg := dma_req_vaddr_next
    dma_req_data_reg := dma_req_data_next
    dma_req_len_reg := dma_req_len_next
    dma_req_mask_reg := dma_req_mask_next
    dma_req_status_reg := dma_req_status_next
  }
  
  // 连接到DMA接口
  io.dmaReq.valid := dma_req_valid_reg
  io.dmaReq.bits.vaddr := dma_req_vaddr_reg
  io.dmaReq.bits.data := dma_req_data_reg
  io.dmaReq.bits.len := dma_req_len_reg
  io.dmaReq.bits.mask := dma_req_mask_reg
  io.dmaReq.bits.status := dma_req_status_reg

  // 连接SRAM响应ready信号 - 基于DMA ready状态
  io.sramRead.foreach(_.resp.ready := io.dmaReq.ready && (state === s_sram_req || state === s_dma_wait))
  io.accRead.foreach(_.resp.ready := io.dmaReq.ready && (state === s_sram_req || state === s_dma_wait))
  // 状态转换和计数器更新
  when (io.sramRead.map(_.req.fire).reduce(_ || _)) {
    state := s_dma_wait
  }

  when (io.dmaReq.fire) {
    when (!final_send) {
      sram_count := sram_count + 1.U
    }
    
    // 更新缓存状态  
    when (addr_offset =/= 0.U && (sram_resp_valid || acc_resp_valid)) {
      // 非对齐情况：缓存新数据的高位部分
      val remaining_bytes = align_bytes.U - addr_offset  // 缓存的是高位部分
      data_buffer := incoming_data >> (addr_offset * 8.U)
      buffer_valid_bytes := remaining_bytes
      // 更新buffer对应的地址（指向下一个16字节对齐地址）
      when (buffer_valid_bytes === 0.U) {
        buffer_start_addr := aligned_addr + align_bytes.U
      }.otherwise {
        buffer_start_addr := buffer_start_addr + align_bytes.U
      }
    }.elsewhen (final_send) {
      // 发送了最后的剩余数据，清空缓存
      buffer_valid_bytes := 0.U
    }.otherwise {
      // 对齐情况下，如果之前有缓存数据被合并发送了，需要清空缓存
      when (buffer_valid_bytes > 0.U && can_send_full_line && sram_resp_valid) {
        buffer_valid_bytes := 0.U
      }
    }
    
    // 修复状态转换逻辑
    when (final_send) {
      // final_send 完成后才回到 idle
      state := s_idle
    }.elsewhen (all_data_sent) {
      // 所有数据都已发送完成
      state := s_idle
    }.elsewhen (sram_count + 1.U >= iter_reg && iter_reg > 0.U) {
      // 迭代结束，但可能还有缓存数据需要发送
      when (buffer_valid_bytes > 0.U) {
        state := s_dma_wait  // 保持状态，等待 final_send
      }.otherwise {
        state := s_idle
      }
    }.elsewhen (iter_reg === 0.U) {
      state := s_idle
    }.otherwise {
      state := s_sram_req
    }
  }

  // 等待DMA真正完成
  io.dmaResp.ready := true.B

  // 修复完成信号逻辑 - 只有在真正完成所有数据传输后才发出完成信号
  val task_complete = RegInit(false.B)
  when (io.cmdReq.fire && io.cmdReq.bits.cmd.mem_decode_cmd.is_store) {
    task_complete := false.B
  }.elsewhen (io.dmaReq.fire && (final_send || all_data_sent)) {
    task_complete := true.B
  }
  
  io.cmdResp.valid := task_complete && (state === s_idle)
  io.cmdResp.bits.rob_id := rob_id_reg
  
  // 发送完成信号后重置标志
  when (io.cmdResp.fire) {
    task_complete := false.B
  }
}
