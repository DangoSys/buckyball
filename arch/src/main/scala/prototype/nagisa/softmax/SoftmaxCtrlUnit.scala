package prototype.nagisa.softmax

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}

class SoftmaxCtrlUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp_o = Decoupled(new BallRsComplete)

    val ctrl_ld_o = Decoupled(new SMCtrlLdReq)
    val ctrl_findmax_o = Decoupled(new SMCtrlFindMaxReq)
    val ctrl_expsum_o = Decoupled(new SMCtrlExpSumReq)
    val ctrl_norm_o = Decoupled(new SMCtrlNormReq)
    val ctrl_st_o = Decoupled(new SMCtrlStReq)

    val cmdResp_i = Flipped(Valid(new Bundle { val commit = Bool() }))
  })

  // Internal registers
  val rob_id_reg    = RegInit(0.U(log2Up(b.rob_entries).W))
  val iter          = RegInit(0.U(10.W))
  val op1_bank      = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val op1_bank_addr = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val wr_bank       = RegInit(0.U(log2Up(b.sp_banks + b.acc_banks).W))
  val wr_bank_addr  = RegInit(0.U(log2Up(b.spad_bank_entries.max(b.acc_bank_entries)).W))
  val is_acc        = RegInit(false.B)

  // Special field registers
  val dim_len       = RegInit(0.U(10.W))
  val batch         = RegInit(0.U(10.W))
  val log_mode      = RegInit(false.B)

  val has_send      = RegInit(false.B)

  // State machine
  val idle :: busy :: Nil = Enum(2)
  val state = RegInit(idle)

  // Capture instruction when cmdReq fires
  when(io.cmdReq.fire) {
    iter          := io.cmdReq.bits.cmd.iter
    rob_id_reg    := io.cmdReq.bits.rob_id
    op1_bank      := io.cmdReq.bits.cmd.op1_bank
    op1_bank_addr := io.cmdReq.bits.cmd.op1_bank_addr
    wr_bank       := io.cmdReq.bits.cmd.wr_bank
    wr_bank_addr  := io.cmdReq.bits.cmd.wr_bank_addr
    is_acc        := io.cmdReq.bits.cmd.is_acc

    // Decode special field
    val special = SoftmaxSpecial.decode(io.cmdReq.bits.cmd.special)
    dim_len     := special.dim_len
    batch       := special.batch
    log_mode    := special.log_mode

    state := busy
  }

  // Send control signals to downstream units
  when(state === busy && !has_send) {
    io.ctrl_ld_o.valid               := true.B
    io.ctrl_ld_o.bits.op1_bank       := op1_bank
    io.ctrl_ld_o.bits.op1_bank_addr  := op1_bank_addr
    io.ctrl_ld_o.bits.iter           := iter
    io.ctrl_ld_o.bits.is_acc         := is_acc
    io.ctrl_ld_o.bits.dim_len        := dim_len
    io.ctrl_ld_o.bits.batch          := batch
    io.ctrl_ld_o.bits.log_mode       := log_mode

    io.ctrl_findmax_o.valid          := true.B
    io.ctrl_findmax_o.bits.iter      := iter
    io.ctrl_findmax_o.bits.dim_len   := dim_len
    io.ctrl_findmax_o.bits.batch     := batch

    io.ctrl_expsum_o.valid           := true.B
    io.ctrl_expsum_o.bits.iter       := iter
    io.ctrl_expsum_o.bits.dim_len    := dim_len
    io.ctrl_expsum_o.bits.batch      := batch

    io.ctrl_norm_o.valid             := true.B
    io.ctrl_norm_o.bits.iter         := iter
    io.ctrl_norm_o.bits.dim_len      := dim_len
    io.ctrl_norm_o.bits.batch        := batch
    io.ctrl_norm_o.bits.log_mode     := log_mode

    io.ctrl_st_o.valid               := true.B
    io.ctrl_st_o.bits.wr_bank        := wr_bank
    io.ctrl_st_o.bits.wr_bank_addr   := wr_bank_addr
    io.ctrl_st_o.bits.iter           := iter
    io.ctrl_st_o.bits.is_acc         := is_acc
    io.ctrl_st_o.bits.dim_len        := dim_len

    // Mark as sent when all control signals are accepted
    when(io.ctrl_ld_o.ready && io.ctrl_findmax_o.ready &&
         io.ctrl_expsum_o.ready && io.ctrl_norm_o.ready &&
         io.ctrl_st_o.ready) {
      has_send := true.B
    }
  }.otherwise {
    io.ctrl_ld_o.valid               := false.B
    io.ctrl_ld_o.bits.op1_bank       := 0.U
    io.ctrl_ld_o.bits.op1_bank_addr  := 0.U
    io.ctrl_ld_o.bits.iter           := 0.U
    io.ctrl_ld_o.bits.is_acc         := false.B
    io.ctrl_ld_o.bits.dim_len        := 0.U
    io.ctrl_ld_o.bits.batch          := 0.U
    io.ctrl_ld_o.bits.log_mode       := false.B

    io.ctrl_findmax_o.valid          := false.B
    io.ctrl_findmax_o.bits.iter      := 0.U
    io.ctrl_findmax_o.bits.dim_len   := 0.U
    io.ctrl_findmax_o.bits.batch     := 0.U

    io.ctrl_expsum_o.valid           := false.B
    io.ctrl_expsum_o.bits.iter       := 0.U
    io.ctrl_expsum_o.bits.dim_len    := 0.U
    io.ctrl_expsum_o.bits.batch      := 0.U

    io.ctrl_norm_o.valid             := false.B
    io.ctrl_norm_o.bits.iter         := 0.U
    io.ctrl_norm_o.bits.dim_len      := 0.U
    io.ctrl_norm_o.bits.batch        := 0.U
    io.ctrl_norm_o.bits.log_mode     := false.B

    io.ctrl_st_o.valid               := false.B
    io.ctrl_st_o.bits.wr_bank        := 0.U
    io.ctrl_st_o.bits.wr_bank_addr   := 0.U
    io.ctrl_st_o.bits.iter           := 0.U
    io.ctrl_st_o.bits.is_acc         := false.B
    io.ctrl_st_o.bits.dim_len        := 0.U
  }

  // Wait for completion signal from store unit
  when(io.cmdResp_i.valid) {
    io.cmdResp_o.valid       := true.B
    io.cmdResp_o.bits.rob_id := rob_id_reg
    state                    := idle
    has_send                 := false.B
  }.otherwise {
    io.cmdResp_o.valid       := false.B
    io.cmdResp_o.bits.rob_id := 0.U
  }

  io.cmdReq.ready := state === idle
}
