package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO, SramReadReq, SramReadResp}
import examples.BuckyBallConfigs.CustomBuckyBallConfig


class ctrl_ld_req(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val op2_bank      = UInt(log2Up(b.sp_banks).W)
  val op2_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val iter          = UInt(10.W)
}

class VecLoadUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  val io = IO(new Bundle {
    val sramReadReq = Vec(b.sp_banks, Decoupled(new SramReadReq(b.spad_bank_entries)))
		val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(b.spad_w))))
    val ctrl_ld_i = Flipped(Decoupled(new ctrl_ld_req))
    val ld_ex_o = Decoupled(new ld_ex_req)
  })

	val op1_bank 		 = RegInit(0.U(log2Up(b.sp_banks).W))
	val op2_bank 		 = RegInit(0.U(log2Up(b.sp_banks).W))
	val op1_addr 		 = RegInit(0.U(log2Up(b.spad_bank_entries).W))
	val op2_addr 		 = RegInit(0.U(log2Up(b.spad_bank_entries).W))
  val iter 				 = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

	// 输出寄存器，用于打破组合逻辑环
	val ld_ex_valid_reg = RegInit(false.B)
	val ld_ex_op1_reg = Reg(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
	val ld_ex_op2_reg = Reg(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
	val ld_ex_iter_reg = RegInit(0.U(10.W))

	// 每个bank读请求默认赋值
  for (i <- 0 until b.sp_banks){
    io.sramReadReq(i).valid 		   := false.B
    io.sramReadReq(i).bits.fromDMA := false.B
    io.sramReadReq(i).bits.addr    := 0.U
  }

	io.ctrl_ld_i.ready := state === idle

// -----------------------------------------------------------------------------
// Ctrl指令到来设置寄存器
// -----------------------------------------------------------------------------

  when (io.ctrl_ld_i.fire) {
		op1_bank 			:= io.ctrl_ld_i.bits.op1_bank
		op2_bank 			:= io.ctrl_ld_i.bits.op2_bank
		op1_addr 			:= io.ctrl_ld_i.bits.op1_bank_addr
		op2_addr 			:= io.ctrl_ld_i.bits.op2_bank_addr
    iter          := io.ctrl_ld_i.bits.iter
		iter_counter 	:= 0.U
    state         := busy
		assert(io.ctrl_ld_i.bits.iter  > 0.U, "iter should be greater than 0")
  }

// -----------------------------------------------------------------------------
// 发送SRAM读请求 (只在输出寄存器空闲时发送)
// -----------------------------------------------------------------------------
	when (state === busy && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
		io.sramReadReq(op1_bank).valid        := true.B
		io.sramReadReq(op1_bank).bits.fromDMA := false.B
		io.sramReadReq(op1_bank).bits.addr    := op1_addr + iter_counter

		io.sramReadReq(op2_bank).valid        := true.B
		io.sramReadReq(op2_bank).bits.fromDMA := false.B
		io.sramReadReq(op2_bank).bits.addr    := op2_addr + iter_counter
		iter_counter 				 									:= iter_counter + 1.U
  }

// -----------------------------------------------------------------------------
// SRAM返回数据, 并传递给EX单元 (使用寄存器打破组合逻辑环)
// -----------------------------------------------------------------------------
	// sramReadResp 的 ready 信号：当没有待发送数据或下游已接收时可以接收
	io.sramReadResp.foreach { resp =>
		resp.ready := !ld_ex_valid_reg || io.ld_ex_o.ready
	}

	// 接收 SRAM 数据并缓存到寄存器
  when (io.sramReadResp(op1_bank).valid && io.sramReadResp(op2_bank).valid &&
        (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
		ld_ex_valid_reg := true.B
    ld_ex_op1_reg := io.sramReadResp(op1_bank).bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
    ld_ex_op2_reg := io.sramReadResp(op2_bank).bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
		ld_ex_iter_reg := iter_counter
  }.elsewhen(io.ld_ex_o.ready) {
		ld_ex_valid_reg := false.B
	}

	// 输出来自寄存器
	io.ld_ex_o.valid := ld_ex_valid_reg
	io.ld_ex_o.bits.op1 := ld_ex_op1_reg
	io.ld_ex_o.bits.op2 := ld_ex_op2_reg
	io.ld_ex_o.bits.iter := ld_ex_iter_reg

	assert((!io.sramReadResp(op1_bank).fire && !io.sramReadResp(op2_bank).fire) ||
				 (io.sramReadResp(op1_bank).fire && io.sramReadResp(op2_bank).fire),
				 "two sramReadResp should be fired in the same time or none of them")


// -----------------------------------------------------------------------------
// iter_counter归零，回归idle状态
// -----------------------------------------------------------------------------

	when(state === busy && iter_counter === iter - 1.U && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
		state 				:= idle
		iter_counter 	:= 0.U
	}

}
