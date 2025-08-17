package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO, SramReadResp}
import examples.toy.balldomain.{ExReservationStationIssue, ExReservationStationComplete}
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import warp.VecBall


class ctrl_ex_req(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val iter = UInt(10.W)
}

class ld_ex_req(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val op1 = Vec(b.veclane, UInt(b.inputType.getWidth.W))
  val op2 = Vec(b.veclane, UInt(b.inputType.getWidth.W))
  val iter = UInt(10.W)
}

class VecEXUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl_ex_i = Flipped(Decoupled(new ctrl_ex_req))
    val ld_ex_i = Flipped(Decoupled(new ld_ex_req))

    val ex_st_o = Decoupled(new ex_st_req)
  })

  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

	val VecBall = Module(new VecBall)

  // Initialize default values for all signals
  io.ctrl_ex_i.ready 		:= false.B
  io.ex_st_o.valid 			:= false.B
  io.ex_st_o.bits.rst 	:= VecInit(Seq.fill(b.veclane)(0.U(b.accType.getWidth.W)))
  io.ex_st_o.bits.iter 	:= 0.U

  // Initialize VecBall input signals with default values
  VecBall.io.iterIn.valid := false.B
  VecBall.io.iterIn.bits 	:= 0.U
  VecBall.io.op1In.valid 	:= false.B
  VecBall.io.op1In.bits 	:= VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
  VecBall.io.op2In.valid 	:= false.B
  VecBall.io.op2In.bits 	:= VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W)))
  VecBall.io.rstOut.ready := false.B

// -----------------------------------------------------------------------------
// Ctrl指令到来设置寄存器
// -----------------------------------------------------------------------------
  io.ctrl_ex_i.ready := state === idle
  when(io.ctrl_ex_i.fire) {
		VecBall.io.iterIn.valid := true.B
		VecBall.io.iterIn.bits  := io.ctrl_ex_i.bits.iter
    state := busy
  }

// -----------------------------------------------------------------------------
// 接受来自load unit的读结果, 并进行计算
// -----------------------------------------------------------------------------
	io.ld_ex_i.ready := state === busy && VecBall.io.iterIn.ready
	when(io.ld_ex_i.valid) {
		VecBall.io.op1In.valid := true.B
		VecBall.io.op1In.bits := io.ld_ex_i.bits.op1
		VecBall.io.op2In.valid := true.B
		VecBall.io.op2In.bits := io.ld_ex_i.bits.op2
		//assert((io.ld_ex_i.bits.iter - VecBall.get_iterCounter() === 16.U) && VecBall.get_arrive(),
					 //"[VecLoad -> VecEX] iteration mismatch")
	}

// -----------------------------------------------------------------------------
// 向store unit发送计算结果，进行写回
// -----------------------------------------------------------------------------
	io.ex_st_o.valid        := VecBall.io.rstOut.valid
	VecBall.io.rstOut.ready := io.ex_st_o.ready

	when(io.ex_st_o.fire) {
		io.ex_st_o.bits.rst   := VecBall.io.rstOut.bits
		io.ex_st_o.bits.iter  := VecBall.io.iterOut.bits
	}

}