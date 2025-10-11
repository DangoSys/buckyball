package prototype.nagisa.layernorm

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}

class LayerNormCtrlUnit(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp_o = Decoupled(new BallRsComplete)

    val ctrl_ld_o = Decoupled(new LNCtrlLdReq)
    val ctrl_reduce_o = Decoupled(new LNCtrlReduceReq)
    val ctrl_norm_o = Decoupled(new LNCtrlNormReq)
    val ctrl_st_o = Decoupled(new LNCtrlStReq)

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
  val norm_dim      = RegInit(0.U(12.W))
  val gamma_addr    = RegInit(0.U(12.W))
  val beta_addr     = RegInit(0.U(12.W))
  val param_bank    = RegInit(0.U(2.W))
  val use_affine    = RegInit(false.B)

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
    val special = LayerNormSpecial.decode(io.cmdReq.bits.cmd.special)
    norm_dim    := special.norm_dim
    gamma_addr  := special.gamma_addr
    beta_addr   := special.beta_addr
    param_bank  := special.param_bank
    use_affine  := special.use_affine

    state := busy
  }

  // Send control signals to downstream units
  when(state === busy && !has_send) {
    io.ctrl_ld_o.valid               := true.B
    io.ctrl_ld_o.bits.op1_bank       := op1_bank
    io.ctrl_ld_o.bits.op1_bank_addr  := op1_bank_addr
    io.ctrl_ld_o.bits.iter           := iter
    io.ctrl_ld_o.bits.is_acc         := is_acc
    io.ctrl_ld_o.bits.norm_dim       := norm_dim
    io.ctrl_ld_o.bits.param_bank     := param_bank
    io.ctrl_ld_o.bits.gamma_addr     := gamma_addr
    io.ctrl_ld_o.bits.beta_addr      := beta_addr
    io.ctrl_ld_o.bits.use_affine     := use_affine

    io.ctrl_reduce_o.valid           := true.B
    io.ctrl_reduce_o.bits.iter       := iter
    io.ctrl_reduce_o.bits.is_acc     := is_acc
    io.ctrl_reduce_o.bits.norm_dim   := norm_dim
    io.ctrl_reduce_o.bits.use_affine := use_affine

    io.ctrl_norm_o.valid             := true.B
    io.ctrl_norm_o.bits.iter         := iter
    io.ctrl_norm_o.bits.is_acc       := is_acc
    io.ctrl_norm_o.bits.norm_dim     := norm_dim
    io.ctrl_norm_o.bits.use_affine   := use_affine

    io.ctrl_st_o.valid               := true.B
    io.ctrl_st_o.bits.wr_bank        := wr_bank
    io.ctrl_st_o.bits.wr_bank_addr   := wr_bank_addr
    io.ctrl_st_o.bits.iter           := iter
    io.ctrl_st_o.bits.is_acc         := is_acc
    io.ctrl_st_o.bits.norm_dim       := norm_dim

    // Mark as sent when all control signals are accepted
    when(io.ctrl_ld_o.ready && io.ctrl_reduce_o.ready &&
         io.ctrl_norm_o.ready && io.ctrl_st_o.ready) {
      has_send := true.B
    }
  }.otherwise {
    io.ctrl_ld_o.valid               := false.B
    io.ctrl_ld_o.bits.op1_bank       := 0.U
    io.ctrl_ld_o.bits.op1_bank_addr  := 0.U
    io.ctrl_ld_o.bits.iter           := 0.U
    io.ctrl_ld_o.bits.is_acc         := false.B
    io.ctrl_ld_o.bits.norm_dim       := 0.U
    io.ctrl_ld_o.bits.param_bank     := 0.U
    io.ctrl_ld_o.bits.gamma_addr     := 0.U
    io.ctrl_ld_o.bits.beta_addr      := 0.U
    io.ctrl_ld_o.bits.use_affine     := false.B

    io.ctrl_reduce_o.valid           := false.B
    io.ctrl_reduce_o.bits.iter       := 0.U
    io.ctrl_reduce_o.bits.is_acc     := false.B
    io.ctrl_reduce_o.bits.norm_dim   := 0.U
    io.ctrl_reduce_o.bits.use_affine := false.B

    io.ctrl_norm_o.valid             := false.B
    io.ctrl_norm_o.bits.iter         := 0.U
    io.ctrl_norm_o.bits.is_acc       := false.B
    io.ctrl_norm_o.bits.norm_dim     := 0.U
    io.ctrl_norm_o.bits.use_affine   := false.B

    io.ctrl_st_o.valid               := false.B
    io.ctrl_st_o.bits.wr_bank        := 0.U
    io.ctrl_st_o.bits.wr_bank_addr   := 0.U
    io.ctrl_st_o.bits.iter           := 0.U
    io.ctrl_st_o.bits.is_acc         := false.B
    io.ctrl_st_o.bits.norm_dim       := 0.U
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
