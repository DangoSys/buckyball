package prototype.nagisa.gelu

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramReadReq, SramReadResp}

class GeluLoadUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val sramReadReq  = Vec(b.sp_banks, Decoupled(new SramReadReq(b.spad_bank_entries)))
    val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(b.spad_w))))
    val accReadReq   = Vec(b.acc_banks, Decoupled(new SramReadReq(b.acc_bank_entries)))
    val accReadResp  = Vec(b.acc_banks, Flipped(Decoupled(new SramReadResp(b.acc_w))))

    val ctrl_ld_i = Flipped(Decoupled(new CtrlLdReq))
    val ld_ex_o   = Decoupled(new LdExReq)
  })

  val op1_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val op1_addr      = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val iter          = RegInit(0.U(10.W))
  val req_cnt       = RegInit(0.U(10.W))  // 已发送的读请求数
  val resp_cnt      = RegInit(0.U(10.W))  // 已接收的读响应数
  val is_acc        = RegInit(false.B)

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

  // 输出寄存器
  val ld_ex_valid_reg = RegInit(false.B)
  val ld_ex_data_reg  = Reg(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
  val ld_ex_is_acc_reg = RegInit(false.B)

  // 默认赋值
  for (i <- 0 until b.sp_banks) {
    io.sramReadReq(i).valid         := false.B
    io.sramReadReq(i).bits.fromDMA  := false.B
    io.sramReadReq(i).bits.addr     := 0.U
  }
  for (i <- 0 until b.acc_banks) {
    io.accReadReq(i).valid          := false.B
    io.accReadReq(i).bits.fromDMA   := false.B
    io.accReadReq(i).bits.addr      := 0.U
  }

  io.ctrl_ld_i.ready := state === idle

  // 控制指令到来
  when(io.ctrl_ld_i.fire) {
    op1_bank  := io.ctrl_ld_i.bits.op1_bank
    op1_addr  := io.ctrl_ld_i.bits.op1_bank_addr
    iter      := io.ctrl_ld_i.bits.iter
    is_acc    := io.ctrl_ld_i.bits.is_acc
    req_cnt   := 0.U
    resp_cnt  := 0.U
    state     := busy
    assert(io.ctrl_ld_i.bits.iter > 0.U, "iter should be greater than 0")
  }

  // 发送读请求
  when(state === busy && req_cnt < iter && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
    when(is_acc) {
      val acc_bank_id = op1_bank - b.sp_banks.U
      io.accReadReq(acc_bank_id).valid       := true.B
      io.accReadReq(acc_bank_id).bits.fromDMA := false.B
      io.accReadReq(acc_bank_id).bits.addr   := op1_addr + req_cnt
      when(io.accReadReq(acc_bank_id).ready) {
        req_cnt := req_cnt + 1.U
      }
    }.otherwise {
      io.sramReadReq(op1_bank).valid         := true.B
      io.sramReadReq(op1_bank).bits.fromDMA  := false.B
      io.sramReadReq(op1_bank).bits.addr     := op1_addr + req_cnt
      when(io.sramReadReq(op1_bank).ready) {
        req_cnt := req_cnt + 1.U
      }
    }
  }

  // 接收读响应
  io.sramReadResp.foreach { resp => resp.ready := !ld_ex_valid_reg || io.ld_ex_o.ready }
  io.accReadResp.foreach { resp => resp.ready := !ld_ex_valid_reg || io.ld_ex_o.ready }

  when(is_acc) {
    val acc_bank_id = op1_bank - b.sp_banks.U
    when(io.accReadResp(acc_bank_id).valid && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      ld_ex_valid_reg  := true.B
      ld_ex_data_reg   := io.accReadResp(acc_bank_id).bits.data.asTypeOf(Vec(b.veclane, UInt(b.accType.getWidth.W)))
      ld_ex_is_acc_reg := is_acc
      resp_cnt         := resp_cnt + 1.U
    }.elsewhen(io.ld_ex_o.ready) {
      ld_ex_valid_reg := false.B
    }
  }.otherwise {
    when(io.sramReadResp(op1_bank).valid && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      ld_ex_valid_reg  := true.B
      ld_ex_data_reg   := io.sramReadResp(op1_bank).bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
      ld_ex_is_acc_reg := is_acc
      resp_cnt         := resp_cnt + 1.U
    }.elsewhen(io.ld_ex_o.ready) {
      ld_ex_valid_reg := false.B
    }
  }

  // 输出
  io.ld_ex_o.valid       := ld_ex_valid_reg
  io.ld_ex_o.bits.data   := ld_ex_data_reg
  io.ld_ex_o.bits.iter   := iter  // 传递迭代次数给EX单元
  io.ld_ex_o.bits.is_acc := ld_ex_is_acc_reg

  // 完成所有迭代后返回idle (所有响应都已接收并发送给下游)
  when(state === busy && resp_cnt === iter && !ld_ex_valid_reg) {
    state := idle
  }
}
