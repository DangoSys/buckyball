package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.rs.{MemRsIssue, MemRsComplete}
import framework.builtin.memdomain.mem.SramWriteIO
import framework.builtin.memdomain.dma.{BBReadRequest, BBReadResponse, LocalAddr}
import freechips.rocketchip.rocket.MStatus

class MemLoader(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)

  val io = IO(new Bundle {
    // Load instruction from ReservationStation
    val cmdReq = Flipped(Decoupled(new MemRsIssue))
    // Completion signal sent to ReservationStation
    val cmdResp = Decoupled(new MemRsComplete)
    // Direct connection to DMA read interface
    val dmaReq = Decoupled(new BBReadRequest())
    val dmaResp = Flipped(Decoupled(new BBReadResponse(b.spad_w)))
    // Connected to Scratchpad SRAM write interface
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  val s_idle :: s_dma_req :: s_dma_wait :: Nil = Enum(3)
  val state = RegInit(s_idle)

  val rob_id_reg = RegInit(0.U(rob_id_width.W))
  // Cache mem_addr
  val mem_addr_reg = Reg(UInt(b.memAddrLen.W))
  // Cache iteration count
  val iter_reg = Reg(UInt(10.W))
  // Count number of responses received, supports up to 16 responses
  val resp_count = Reg(UInt(log2Up(16).W))

  // Cache decoded bank information
  val wr_bank_reg = Reg(UInt(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr_reg = Reg(UInt(log2Up(b.spad_bank_entries).W))
  // Whether this is an acc bank operation
  val is_acc_reg = RegInit(false.B)
  // Cache stride
  val stride_reg = Reg(UInt(10.W))

  // Receive load instruction
  io.cmdReq.ready := state === s_idle

  when (io.cmdReq.fire && io.cmdReq.bits.cmd.is_load) {
    state              := s_dma_req
    rob_id_reg         := io.cmdReq.bits.rob_id
    mem_addr_reg       := io.cmdReq.bits.cmd.mem_addr
    iter_reg           := io.cmdReq.bits.cmd.iter
    wr_bank_reg        := io.cmdReq.bits.cmd.sp_bank
    wr_bank_addr_reg   := io.cmdReq.bits.cmd.sp_bank_addr
    // Determine if acc based on bank
    is_acc_reg         := (io.cmdReq.bits.cmd.sp_bank >= b.sp_banks.U)
    stride_reg         := io.cmdReq.bits.cmd.special(10,0)
    resp_count         := 0.U
  }

  // Issue DMA read request - read iter_reg rows of data
  io.dmaReq.valid       := state === s_dma_req
  io.dmaReq.bits.vaddr  := mem_addr_reg
  // Byte count of iter rows of data
  io.dmaReq.bits.len    := iter_reg * (b.veclane * b.inputType.getWidth / 8).U
  // Simplified: use default status
  io.dmaReq.bits.status := 0.U.asTypeOf(new MStatus)
  io.dmaReq.bits.stride := stride_reg

  when (io.dmaReq.fire) {
    state := s_dma_wait
    // Reset response counter
    resp_count := 0.U
  }

  // Wait for DMA response
  io.dmaResp.ready := state === s_dma_wait

  when (io.dmaResp.fire) {
    resp_count := resp_count + 1.U
    // Return to idle state when last response is received
    when (io.dmaResp.bits.last) {
      state := s_idle
    }
  }

  // Stream write to SRAM - write immediately upon receiving each response
  // Calculate current write bank and address
  // Use address counter from DMA response
  val current_bank_addr = wr_bank_addr_reg + io.dmaResp.bits.addrcounter
  // All responses write to the same bank
  val target_bank = wr_bank_reg
  val target_row = current_bank_addr

  for (i <- 0 until b.sp_banks) {
    io.sramWrite(i).req.valid     := io.dmaResp.fire && (target_bank === i.U)
    io.sramWrite(i).req.bits.addr := target_row
    io.sramWrite(i).req.bits.data := io.dmaResp.bits.data
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(true.B))
  }
  // Default assignment
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits.addr := 0.U
    io.accWrite(i).req.bits.data := 0.U
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(false.B))
  }

  for (i <- 0 until b.acc_banks/2) {
    when(io.dmaResp.fire && is_acc_reg){
      when(io.dmaResp.bits.addrcounter(2)){
        io.accWrite(i).req.valid     := target_row(log2Ceil(b.acc_banks/2) - 1, 0) === i.U
        io.accWrite(i).req.bits.addr := wr_bank_addr_reg + (io.dmaResp.bits.addrcounter >> (log2Ceil(b.acc_banks/2) + 1))
        io.accWrite(i).req.bits.data := io.dmaResp.bits.data
        io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }.otherwise{
        io.accWrite(i + b.acc_banks/2).req.valid     := target_row(log2Ceil(b.acc_banks/2) - 1, 0) === i.U
        io.accWrite(i + b.acc_banks/2).req.bits.addr := wr_bank_addr_reg + (io.dmaResp.bits.addrcounter >>  (log2Ceil(b.acc_banks/2) + 1))
        io.accWrite(i + b.acc_banks/2).req.bits.data := io.dmaResp.bits.data
        io.accWrite(i + b.acc_banks/2).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
      }
    }
  }

  // Send completion signal - only send when last response is received
  io.cmdResp.valid := io.dmaResp.fire && io.dmaResp.bits.last
  io.cmdResp.bits.rob_id := rob_id_reg
}
