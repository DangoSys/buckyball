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
	val spad_w = b.veclane * b.inputType.getWidth
  val io = IO(new Bundle {
    val sramReadReq = Vec(b.sp_banks, Decoupled(new SramReadReq(b.spad_bank_entries)))
		val sramReadResp = Vec(b.sp_banks, Flipped(Decoupled(new SramReadResp(spad_w))))
    val ctrl_ld_i = Flipped(Decoupled(new ctrl_ld_req))
    val ld_ex_o = Decoupled(new ld_ex_req)
  })

	// val op1_reg = RegInit(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W))))
	// val op2_reg = RegInit(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W))))

	val op1_bank 		 = RegInit(0.U(log2Up(b.sp_banks).W))
	val op2_bank 		 = RegInit(0.U(log2Up(b.sp_banks).W))
	val op1_addr 		 = RegInit(0.U(log2Up(b.spad_bank_entries).W))
	val op2_addr 		 = RegInit(0.U(log2Up(b.spad_bank_entries).W))
  val iter 				 = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

	//每个bank读请求默认赋值
  for(i <- 0 until b.sp_banks){
    io.sramReadReq(i).valid 		   := false.B
    io.sramReadReq(i).bits.fromDMA := false.B
    io.sramReadReq(i).bits.addr    := 0.U
  }

	io.ctrl_ld_i.ready := state === idle

// -----------------------------------------------------------------------------
// Ctrl指令到来设置寄存器
// -----------------------------------------------------------------------------

  when(io.ctrl_ld_i.fire) {
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
// SRAM返回数据寄存器定义（需要在使用前定义）
// -----------------------------------------------------------------------------
	// 使用寄存器缓冲SRAM响应数据，避免组合逻辑环路
	val sram_data_valid = RegNext(io.sramReadResp(op1_bank).valid && io.sramReadResp(op2_bank).valid, false.B)
	val sram_op1_data = RegEnable(io.sramReadResp(op1_bank).bits.data, io.sramReadResp(op1_bank).valid)
	val sram_op2_data = RegEnable(io.sramReadResp(op2_bank).bits.data, io.sramReadResp(op2_bank).valid)
	val sram_iter_data = RegEnable(iter_counter, io.sramReadResp(op1_bank).valid && io.sramReadResp(op2_bank).valid)

// -----------------------------------------------------------------------------
// 发送SRAM读请求
// -----------------------------------------------------------------------------
	// 只有当下游准备好接收数据且没有待处理的SRAM请求时才发送新请求
	val can_send_req = state === busy && io.ld_ex_o.ready && !sram_data_valid
	when(can_send_req) {
		io.sramReadReq(op1_bank).valid        := true.B
		io.sramReadReq(op1_bank).bits.fromDMA := false.B
		io.sramReadReq(op1_bank).bits.addr    := op1_addr + iter_counter

		io.sramReadReq(op2_bank).valid        := true.B
		io.sramReadReq(op2_bank).bits.fromDMA := false.B
		io.sramReadReq(op2_bank).bits.addr    := op2_addr + iter_counter

		// 只在成功发送请求时递增计数器
		when(io.sramReadReq(op1_bank).ready && io.sramReadReq(op2_bank).ready) {
			iter_counter := iter_counter + 1.U
		}
  }

// -----------------------------------------------------------------------------
// SRAM返回数据, 并传递给EX单元
// -----------------------------------------------------------------------------

	// SRAM响应总是准备好接收数据
	io.sramReadResp.foreach { resp =>
		resp.ready := true.B
	}

	// 使用注册的数据驱动输出
  when(sram_data_valid) {
		io.ld_ex_o.valid 		 := true.B
    io.ld_ex_o.bits.op1  := sram_op1_data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
    io.ld_ex_o.bits.op2  := sram_op2_data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
		io.ld_ex_o.bits.iter := sram_iter_data
  }.otherwise {
		io.ld_ex_o.valid 		 := false.B
		io.ld_ex_o.bits.op1  := VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
		io.ld_ex_o.bits.op2  := VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
		io.ld_ex_o.bits.iter := 0.U
	}

	assert((!io.sramReadResp(op1_bank).fire && !io.sramReadResp(op2_bank).fire) ||
				 (io.sramReadResp(op1_bank).fire && io.sramReadResp(op2_bank).fire),
				 "two sramReadResp should be fired in the same time or none of them")


// -----------------------------------------------------------------------------
// iter_counter归零，回归idle状态
// -----------------------------------------------------------------------------

	when(state === busy && iter_counter === iter && io.ld_ex_o.fire && !sram_data_valid) {
		state 				:= idle
		iter_counter 	:= 0.U
	}

}
