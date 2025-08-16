package framework.builtin.load

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.memdomain.{MemReservationStationIssue, MemReservationStationComplete, MemBuckyBallCmd}
import framework.builtin.mem.{SimpleReadRequest, SimpleReadResponse, SramWriteIO}
import framework.builtin.frontend.FrontendTLBIO
import freechips.rocketchip.rocket.MStatus

class MemLoader(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  
  val io = IO(new Bundle {
    // 来自ReservationStation的load指令
    val cmdReq = Flipped(Decoupled(new MemReservationStationIssue))
    // 发送给ReservationStation的完成信号
    val cmdResp = Decoupled(new MemReservationStationComplete)
    // 直接连接DMA读取接口
    val dmaReq = Decoupled(new SimpleReadRequest())
    val dmaResp = Flipped(Decoupled(new SimpleReadResponse(b.spad_w)))
    // 连接到Scratchpad的SRAM写入接口
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  val s_idle :: s_dma_req :: s_dma_wait :: Nil = Enum(3)
  val state = RegInit(s_idle)
  
  val rob_id_reg = RegInit(0.U(rob_id_width.W))
  val mem_addr_reg = Reg(UInt(b.memAddrLen.W))  // 缓存mem_addr
  val iter_reg = Reg(UInt(10.W))  // 缓存迭代次数
  val resp_count = Reg(UInt(log2Up(16).W))  // 计数接收到的响应数量，最多支持16个响应
  
  // 缓存解码好的bank信息
  val wr_bank_reg = Reg(UInt(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr_reg = Reg(UInt(log2Up(b.spad_bank_entries).W))
  val is_acc_reg = RegInit(false.B) // 是否是acc bank的操作

  // 接收load指令
  io.cmdReq.ready := state === s_idle
  
  when (io.cmdReq.fire && io.cmdReq.bits.cmd.mem_decode_cmd.is_load) {
    state              := s_dma_req
    rob_id_reg         := io.cmdReq.bits.rob_id
    mem_addr_reg       := io.cmdReq.bits.cmd.mem_decode_cmd.mem_addr
    iter_reg           := io.cmdReq.bits.cmd.mem_decode_cmd.iter
    wr_bank_reg        := io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank
    wr_bank_addr_reg   := io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank_addr
    is_acc_reg         := (io.cmdReq.bits.cmd.mem_decode_cmd.sp_bank >= b.sp_banks.U) // 根据bank判断是否是acc
    resp_count         := 0.U
  }

  // 发起DMA读取请求 - 读取iter_reg行数据
  io.dmaReq.valid       := state === s_dma_req
  io.dmaReq.bits.vaddr  := mem_addr_reg
  io.dmaReq.bits.len    := iter_reg * (b.veclane * b.inputType.getWidth / 8).U // iter行数据的字节数
  io.dmaReq.bits.status := 0.U.asTypeOf(new MStatus) // 简化：使用默认状态

  when (io.dmaReq.fire) {
    state := s_dma_wait
    resp_count := 0.U  // 重置响应计数器
  }

  // 等待DMA响应
  io.dmaResp.ready := state === s_dma_wait
  
  when (io.dmaResp.fire) {
    resp_count := resp_count + 1.U
    // 收到最后一个响应时转回idle状态
    when (io.dmaResp.bits.last) {
      state := s_idle
    }
  }

  // 流式写入SRAM - 每收到一个响应就立即写入
  // 计算当前写入的bank和地址
  val current_bank_addr = wr_bank_addr_reg + io.dmaResp.bits.addrcounter // 使用DMA响应中的地址计数器
  val target_bank = wr_bank_reg  // 所有响应都写入同一个bank
  val target_row = current_bank_addr
  
  for (i <- 0 until b.sp_banks) {
    io.sramWrite(i).req.valid     := io.dmaResp.fire && (target_bank === i.U)
    io.sramWrite(i).req.bits.addr := target_row
    io.sramWrite(i).req.bits.data := io.dmaResp.bits.data
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(true.B))
  }
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid     := io.dmaResp.fire && is_acc_reg && (target_row(log2Ceil(b.acc_banks) - 1, 0) === i.U)
    io.accWrite(i).req.bits.addr := wr_bank_addr_reg + (io.dmaResp.bits.addrcounter >> log2Ceil(b.acc_banks))
    io.accWrite(i).req.bits.data := io.dmaResp.bits.data
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
  }

  // 发送完成信号 - 只有收到最后一个响应时才发送
  io.cmdResp.valid := io.dmaResp.fire && io.dmaResp.bits.last
  io.cmdResp.bits.rob_id := rob_id_reg
}
