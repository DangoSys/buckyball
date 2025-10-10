package prototype.nagisa.gelu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.SramWriteIO

class GeluStoreUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_st_i = Flipped(Decoupled(new CtrlStReq))
    val ex_st_i   = Flipped(Decoupled(new ExStReq))

    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    val cmdResp_o = Valid(new Bundle { val commit = Bool() })
  })

  val wr_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter         = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))
  val is_acc       = RegInit(false.B)

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

  // 控制指令到来
  io.ctrl_st_i.ready := state === idle

  when(io.ctrl_st_i.fire) {
    wr_bank      := io.ctrl_st_i.bits.wr_bank
    wr_bank_addr := io.ctrl_st_i.bits.wr_bank_addr
    iter         := io.ctrl_st_i.bits.iter
    is_acc       := io.ctrl_st_i.bits.is_acc
    iter_counter := 0.U
    state        := busy
  }

  // 默认值
  io.sramWrite.foreach { w =>
    w.req.valid := false.B
    w.req.bits.addr := 0.U
    w.req.bits.data := 0.U
    w.req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(false.B))
  }

  io.accWrite.foreach { w =>
    w.req.valid := false.B
    w.req.bits.addr := 0.U
    w.req.bits.data := 0.U
    w.req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(false.B))
  }

  // 接收EX的结果并写回
  io.ex_st_i.ready := state === busy

  when(io.ex_st_i.fire) {
    // 捕获从EX单元传来的迭代信息
    when(iter_counter === 0.U) {
      iter := io.ex_st_i.bits.iter  // 第一次接收数据时更新iter
    }
  }

  // SRAM写请求逻辑 - 当EX数据有效时写入
  when(state === busy && io.ex_st_i.valid) {
    when(is_acc) {
      // 写入Accumulator
      val acc_bank_id = wr_bank - b.sp_banks.U
      io.accWrite(acc_bank_id).req.valid := true.B
      io.accWrite(acc_bank_id).req.bits.addr := wr_bank_addr + iter_counter
      io.accWrite(acc_bank_id).req.bits.data := Cat(io.ex_st_i.bits.data.reverse)
      io.accWrite(acc_bank_id).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(true.B))
    }.otherwise {
      // 写入Scratchpad (需要转换INT32到INT8)
      val int8_data = Wire(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
      for (i <- 0 until b.veclane) {
        // 简单截断高位，实际应该做饱和转换
        int8_data(i) := io.ex_st_i.bits.data(i)(b.inputType.getWidth - 1, 0)
      }
      io.sramWrite(wr_bank).req.valid := true.B
      io.sramWrite(wr_bank).req.bits.addr := wr_bank_addr + iter_counter
      io.sramWrite(wr_bank).req.bits.data := Cat(int8_data.reverse)
      io.sramWrite(wr_bank).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(true.B))
    }
  }.otherwise {
    // 没有EX数据时，清除写请求
    when(is_acc) {
      val acc_bank_id = wr_bank - b.sp_banks.U
      io.accWrite(acc_bank_id).req.valid := false.B
    }.otherwise {
      io.sramWrite(wr_bank).req.valid := false.B
    }
  }

  // 只有当EX数据被接收时才增加计数
  when(io.ex_st_i.fire) {
    iter_counter := iter_counter + 1.U
  }

  // 完成条件：所有迭代完成且没有待处理的EX数据
  val all_iterations_complete = iter_counter === iter && iter =/= 0.U
  val no_pending_ex_data = !io.ex_st_i.valid
  val all_done = all_iterations_complete && no_pending_ex_data

  when(state === busy && all_done) {
    state := idle
    io.cmdResp_o.valid := true.B
    io.cmdResp_o.bits.commit := true.B
  }.otherwise {
    io.cmdResp_o.valid := false.B
    io.cmdResp_o.bits.commit := false.B
  }
}
