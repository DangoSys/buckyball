package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayCtrl(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp_o = Decoupled(new BallRsComplete(b))

    val ctrl_ld_o = Decoupled(new ctrl_ld_req(b))
    val ctrl_st_o = Decoupled(new ctrl_st_req(b))
    val ctrl_ex_o = Decoupled(new ctrl_ex_req(b))

    val cmdResp_i = Flipped(Valid(new Bundle { val commit = Bool() }))
  })

  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val iter           = RegInit(0.U(b.frontend.iter_len.W))
  val op1_bank       = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op1_bank_addr  = RegInit(0.U(12.W))
  val op2_bank       = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2_bank_addr  = RegInit(0.U(12.W))
  val wr_bank        = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank_addr   = RegInit(0.U(12.W))
  val has_send       = RegInit(false.B)

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  io.cmdReq.ready := state === idle

  when(io.cmdReq.fire) {
    iter           := io.cmdReq.bits.cmd.iter
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
    op1_bank       := io.cmdReq.bits.cmd.op1_bank
    op1_bank_addr  := 0.U
    op2_bank       := io.cmdReq.bits.cmd.op2_bank
    op2_bank_addr  := 0.U
    wr_bank        := io.cmdReq.bits.cmd.wr_bank
    wr_bank_addr   := 0.U
    state          := busy
  }

  when(state === busy && !has_send) {
    io.ctrl_ld_o.valid              := true.B
    io.ctrl_ld_o.bits.op1_bank      := op1_bank
    io.ctrl_ld_o.bits.op1_bank_addr := op1_bank_addr
    io.ctrl_ld_o.bits.op2_bank      := op2_bank
    io.ctrl_ld_o.bits.op2_bank_addr := op2_bank_addr
    io.ctrl_ld_o.bits.iter          := iter

    io.ctrl_ex_o.valid     := true.B
    io.ctrl_ex_o.bits.iter := iter

    io.ctrl_st_o.valid             := true.B
    io.ctrl_st_o.bits.wr_bank      := wr_bank
    io.ctrl_st_o.bits.wr_bank_addr := wr_bank_addr
    io.ctrl_st_o.bits.iter         := iter

    has_send := true.B
  }.otherwise {
    io.ctrl_ld_o.valid              := false.B
    io.ctrl_ld_o.bits.op1_bank      := 0.U
    io.ctrl_ld_o.bits.op1_bank_addr := 0.U
    io.ctrl_ld_o.bits.op2_bank      := 0.U
    io.ctrl_ld_o.bits.op2_bank_addr := 0.U
    io.ctrl_ld_o.bits.iter          := 0.U

    io.ctrl_ex_o.valid     := false.B
    io.ctrl_ex_o.bits.iter := 0.U

    io.ctrl_st_o.valid             := false.B
    io.ctrl_st_o.bits.wr_bank      := 0.U
    io.ctrl_st_o.bits.wr_bank_addr := 0.U
    io.ctrl_st_o.bits.iter         := 0.U
  }

  when(io.cmdResp_i.valid) {
    io.cmdResp_o.valid           := true.B
    io.cmdResp_o.bits.rob_id     := rob_id_reg
    io.cmdResp_o.bits.is_sub     := is_sub_reg
    io.cmdResp_o.bits.sub_rob_id := sub_rob_id_reg
    state                        := idle
    has_send                     := false.B
  }.otherwise {
    io.cmdResp_o.valid           := false.B
    io.cmdResp_o.bits.rob_id     := 0.U
    io.cmdResp_o.bits.is_sub     := false.B
    io.cmdResp_o.bits.sub_rob_id := 0.U
  }
}
