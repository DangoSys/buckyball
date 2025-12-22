package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam

@instantiable
class VecCtrlUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(parameter)))
    val cmdResp_o = Decoupled(new BallRsComplete(parameter))

    val ctrl_ld_o = Decoupled(new ctrl_ld_req(parameter))
    val ctrl_st_o = Decoupled(new ctrl_st_req(parameter))
    val ctrl_ex_o = Decoupled(new ctrl_ex_req)

    val cmdResp_i = Flipped(Valid(new Bundle { val commit = Bool() })) // from store unit
  })

  val rob_id_reg    = RegInit(0.U(log2Up(parameter.rob_entries).W))
  val iter          = RegInit(0.U(10.W))
  val op1_bank      = RegInit(0.U(log2Up(parameter.numBanks).W))
  val op1_bank_addr = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val op2_bank_addr = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val op2_bank      = RegInit(0.U(log2Up(parameter.numBanks).W))
  val wr_bank       = RegInit(0.U(log2Up(parameter.numBanks).W))
  val wr_bank_addr  = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val is_acc        = RegInit(false.B)   // Deprecated: use wmode instead
  val has_send      = RegInit(false.B)
  val mode          = RegInit(0.U(1.W))

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

// -----------------------------------------------------------------------------
// Set registers when EX instruction arrives
// -----------------------------------------------------------------------------

  when(io.cmdReq.fire) {
    iter          := io.cmdReq.bits.cmd.iter
    rob_id_reg    := io.cmdReq.bits.rob_id
    op1_bank      := io.cmdReq.bits.cmd.op1_bank
    op1_bank_addr := 0.U     // New ISA: all operations start from row 0
    op2_bank      := io.cmdReq.bits.cmd.op2_bank
    op2_bank_addr := 0.U     // New ISA: all operations start from row 0
    wr_bank       := io.cmdReq.bits.cmd.wr_bank
    wr_bank_addr  := 0.U     // New ISA: all operations start from row 0
    is_acc        := false.B // Deprecated: use wmode instead
    mode          := io.cmdReq.bits.cmd.special(0)

    state := busy
  }

// -----------------------------------------------------------------------------
// Send control signals to VecUnit's load/store/ex units
// -----------------------------------------------------------------------------

  when(state === busy && !has_send) {
    io.ctrl_ld_o.valid              := true.B
    io.ctrl_ld_o.bits.op1_bank      := op1_bank
    io.ctrl_ld_o.bits.op1_bank_addr := op1_bank_addr
    io.ctrl_ld_o.bits.op2_bank      := op2_bank
    io.ctrl_ld_o.bits.op2_bank_addr := op2_bank_addr
    io.ctrl_ld_o.bits.iter          := iter
    io.ctrl_ld_o.bits.mode          := mode

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
    io.ctrl_ld_o.bits.mode          := 0.U

    io.ctrl_ex_o.valid     := false.B
    io.ctrl_ex_o.bits.iter := 0.U

    io.ctrl_st_o.valid             := false.B
    io.ctrl_st_o.bits.wr_bank      := 0.U
    io.ctrl_st_o.bits.wr_bank_addr := 0.U
    io.ctrl_st_o.bits.iter         := 0.U
  }

// -----------------------------------------------------------------------------
// Wait for VecUnit's final write-back to complete
// -----------------------------------------------------------------------------

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
