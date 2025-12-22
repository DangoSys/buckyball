package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.MemDomainParam
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import framework.memdomain.backend.banks.SramWriteIO
import framework.memdomain.frontend.outside_channel.dma.{BBReadRequest, BBReadResponse}
import freechips.rocketchip.rocket.MStatus
import framework.balldomain.blink.BankWrite
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

@instantiable
class MemLoader(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {
  val rob_id_width = log2Up(parameter.rob_entries)

  @public
  val io = IO(new Bundle {
    // Load instruction from ReservationStation
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(parameter)))
    // Completion signal sent to ReservationStation
    val cmdResp = Decoupled(new MemRsComplete(parameter))
    // Direct connection to DMA read interface
    val dmaReq  = Decoupled(new BBReadRequest())
    val dmaResp = Flipped(Decoupled(new BBReadResponse(parameter.bankWidth)))

    // Connected to Bank write interface
    val bankWrite = Vec(
      parameter.bankNum,
      Flipped(new BankWrite(
        parameter.bankEntries,
        parameter.bankWidth,
        parameter.bankMaskLen,
        parameter.rob_entries,
        parameter.bankNum
      ))
    )

  })

  val s_idle :: s_dma_req :: s_dma_wait :: Nil = Enum(3)
  val state                                    = RegInit(s_idle)

  val rob_id_reg   = RegInit(0.U(rob_id_width.W))
  // Cache mem_addr
  val mem_addr_reg = Reg(UInt(parameter.memAddrLen.W))
  // Cache iteration count
  val iter_reg     = Reg(UInt(10.W))
  // Count number of responses received, supports up to 16 responses
  val resp_count   = Reg(UInt(log2Up(16).W))

  // Cache decoded bank information
  val wr_bank_reg = Reg(UInt(log2Up(parameter.bankNum).W))
  // Cache stride
  val stride_reg  = Reg(UInt(10.W))

  // Receive load instruction
  io.cmdReq.ready := state === s_idle

  when(io.cmdReq.fire && io.cmdReq.bits.cmd.is_load) {
    state        := s_dma_req
    rob_id_reg   := io.cmdReq.bits.rob_id
    mem_addr_reg := io.cmdReq.bits.cmd.mem_addr
    iter_reg     := io.cmdReq.bits.cmd.iter
    wr_bank_reg  := io.cmdReq.bits.cmd.bank_id
    stride_reg   := io.cmdReq.bits.cmd.special(10, 0)
    resp_count   := 0.U
  }

  // Issue DMA read request - read iter_reg rows of data
  io.dmaReq.valid       := state === s_dma_req
  io.dmaReq.bits.vaddr  := mem_addr_reg
  // Byte count of iter rows of data
  io.dmaReq.bits.len    := iter_reg * (parameter.bankWidth / 8).U
  // Simplified: use default status
  io.dmaReq.bits.status := 0.U.asTypeOf(new MStatus)
  io.dmaReq.bits.stride := stride_reg

  when(io.dmaReq.fire) {
    state      := s_dma_wait
    // Reset response counter
    resp_count := 0.U
  }

  // Wait for DMA response
  io.dmaResp.ready := state === s_dma_wait

  when(io.dmaResp.fire) {
    resp_count := resp_count + 1.U
    // Return to idle state when last response is received
    when(io.dmaResp.bits.last) {
      state := s_idle
    }
  }

  // Stream write to SRAM - write immediately upon receiving each response
  // All responses write to the same bank, starting from row 0
  val target_bank = wr_bank_reg
  val target_row  = io.dmaResp.bits.addrcounter

  for (i <- 0 until parameter.bankNum) {
    io.bankWrite(i).io.req.valid      := io.dmaResp.fire && (target_bank === i.U)
    io.bankWrite(i).io.req.bits.addr  := target_row
    io.bankWrite(i).io.req.bits.data  := io.dmaResp.bits.data
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(parameter.bankMaskLen)(true.B))
    io.bankWrite(i).io.req.bits.wmode := false.B // Load is always overwrite
    io.bankWrite(i).rob_id            := rob_id_reg
    io.bankWrite(i).bank_id           := target_bank
  }

  // Send completion signal - only send when last response is received
  io.cmdResp.valid       := io.dmaResp.fire && io.dmaResp.bits.last
  io.cmdResp.bits.rob_id := rob_id_reg
}
