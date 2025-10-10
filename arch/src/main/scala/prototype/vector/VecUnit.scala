package prototype.vector
import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.Status


class VecUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // 连接到Scratchpad的SRAM读写接口
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    // val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))
    // 连接到Accumulator的读写接口
    // val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    // Status output
    val status = new Status
  })

// -----------------------------------------------------------------------------
// VECCTRLUNIT
// -----------------------------------------------------------------------------
  val VecCtrlUnit = Module(new VecCtrlUnit)
  VecCtrlUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// -----------------------------------------------------------------------------
// VECLOADUNIT
// -----------------------------------------------------------------------------
	val VecLoadUnit = Module(new VecLoadUnit)
	VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o
	for (i <- 0 until b.sp_banks) {
		io.sramRead(i).req <> VecLoadUnit.io.sramReadReq(i)
		VecLoadUnit.io.sramReadResp(i) <> io.sramRead(i).resp
	}

// -----------------------------------------------------------------------------
// VECEX
// -----------------------------------------------------------------------------
	val VecEX = Module(new VecEXUnit)
	VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
	VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o


// -----------------------------------------------------------------------------
// VECSTOREUNIT
// -----------------------------------------------------------------------------
	val VecStoreUnit = Module(new VecStoreUnit)
	VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
  VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
	for (i <- 0 until b.acc_banks) {
		io.accWrite(i) <> VecStoreUnit.io.accWrite(i)
	}
	VecCtrlUnit.io.cmdResp_i <> VecStoreUnit.io.cmdResp_o


// -----------------------------------------------------------------------------
// Set DontCare
// -----------------------------------------------------------------------------
  // for (i <- 0 until b.sp_banks) {
  //   io.sramWrite(i) := DontCare
  // }
  // for (i <- 0 until b.acc_banks) {
  //   io.accRead(i) := DontCare
  // }

// -----------------------------------------------------------------------------
// Status tracking
// -----------------------------------------------------------------------------
  val iterCnt = RegInit(0.U(32.W))
  val hasInput = RegInit(false.B)
  val hasOutput = RegInit(false.B)

  when(io.cmdReq.fire) {
    hasInput := true.B
  }
  when(io.cmdResp.fire) {
    hasOutput := false.B
    hasInput := false.B
    iterCnt := iterCnt + 1.U
  }
  when(io.cmdResp.valid && !hasOutput) {
    hasOutput := true.B
  }

  io.status.ready := io.cmdReq.ready
  io.status.valid := io.cmdResp.valid
  io.status.idle := !hasInput && !hasOutput
  io.status.init := hasInput && !hasOutput
  io.status.running := hasOutput
  io.status.complete := io.cmdResp.fire
  io.status.iter := iterCnt
}
