package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.blink.Status

class SoftmaxUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Connect to Scratchpad SRAM read/write interface
    val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    // Connect to Accumulator read/write interface
    val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    // Status output
    val status = new Status
  })

  // Instantiate sub-units
  val ctrlUnit = Module(new SoftmaxCtrlUnit)
  val loadUnit = Module(new SoftmaxLoadUnit)
  val findMaxUnit = Module(new SoftmaxFindMaxUnit)
  val expSumUnit = Module(new SoftmaxExpSumUnit)
  val normalizeUnit = Module(new SoftmaxNormalizeUnit)
  val storeUnit = Module(new SoftmaxStoreUnit)

  // Connect Control Unit
  ctrlUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> ctrlUnit.io.cmdResp_o

  // Connect Load Unit
  loadUnit.io.ctrl_ld_i <> ctrlUnit.io.ctrl_ld_o
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req <> loadUnit.io.sramReadReq(i)
    loadUnit.io.sramReadResp(i) <> io.sramRead(i).resp
  }
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).req <> loadUnit.io.accReadReq(i)
    loadUnit.io.accReadResp(i) <> io.accRead(i).resp
  }

  // Connect FindMax Unit
  findMaxUnit.io.ctrl_findmax_i <> ctrlUnit.io.ctrl_findmax_o
  findMaxUnit.io.ld_findmax_i <> loadUnit.io.ld_findmax_o

  // Connect ExpSum Unit
  expSumUnit.io.ctrl_expsum_i <> ctrlUnit.io.ctrl_expsum_o
  expSumUnit.io.findmax_expsum_i <> findMaxUnit.io.findmax_expsum_o
  expSumUnit.io.findmax_data_i <> findMaxUnit.io.findmax_norm_o

  // Connect Normalize Unit
  normalizeUnit.io.ctrl_norm_i <> ctrlUnit.io.ctrl_norm_o
  normalizeUnit.io.expsum_norm_i <> expSumUnit.io.expsum_norm_o
  normalizeUnit.io.expsum_data_i <> expSumUnit.io.expsum_data_o

  // Connect Store Unit
  storeUnit.io.ctrl_st_i <> ctrlUnit.io.ctrl_st_o
  storeUnit.io.norm_st_i <> normalizeUnit.io.norm_st_o
  for (i <- 0 until b.sp_banks) {
    io.sramWrite(i) <> storeUnit.io.sramWrite(i)
  }
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i) <> storeUnit.io.accWrite(i)
  }
  ctrlUnit.io.cmdResp_i <> storeUnit.io.cmdResp_o

  // Status tracking
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
