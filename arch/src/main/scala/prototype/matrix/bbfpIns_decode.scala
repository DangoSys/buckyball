package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class BBFP_ID(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val rob_id_width = log2Up(b.rob_entries)
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle{
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val is_matmul_ws  = Output(Bool())
    val id_lu_o = Decoupled(new id_lu_req)
  })

  val idle :: busy :: Nil = Enum(2)
  // Register definitions
  val state = RegInit(idle)
  val rob_id_reg = RegInit(0.U(rob_id_width.W))
  val iteration_counter = RegInit(0.U(10.W))
  val iteration = RegInit(0.U(10.W))
  val op1_bank = RegInit(0.U(2.W))
  val op1_bank_addr = RegInit(0.U(12.W))
  val op2_bank_addr = RegInit(0.U(12.W))
  val op2_bank = RegInit(0.U(2.W))
  val wr_bank = RegInit(0.U(2.W))
  val wr_bank_addr = RegInit(0.U(12.W))
  val is_matmul_ws = RegInit(false.B)
  io.is_matmul_ws := false.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.valid && io.cmdReq.bits.cmd.bid === 1.U) {
        iteration         := io.cmdReq.bits.cmd.iter
        iteration_counter := 0.U
        is_matmul_ws      := false.B
        rob_id_reg        := io.cmdReq.bits.rob_id
        op1_bank          := io.cmdReq.bits.cmd.op1_bank
        op1_bank_addr     := io.cmdReq.bits.cmd.op1_bank_addr
        op2_bank          := io.cmdReq.bits.cmd.op2_bank
        op2_bank_addr     := io.cmdReq.bits.cmd.op2_bank_addr
        wr_bank           := io.cmdReq.bits.cmd.wr_bank
        wr_bank_addr      := io.cmdReq.bits.cmd.wr_bank_addr
        state             := busy
        io.is_matmul_ws   := false.B
      }
      when(io.cmdReq.valid && io.cmdReq.bits.cmd.special(0)){
        iteration         := io.cmdReq.bits.cmd.iter
        iteration_counter := 0.U
        is_matmul_ws      := true.B
        rob_id_reg        := io.cmdReq.bits.rob_id
        op1_bank          := io.cmdReq.bits.cmd.op1_bank
        op1_bank_addr     := io.cmdReq.bits.cmd.op1_bank_addr
        op2_bank          := io.cmdReq.bits.cmd.op2_bank
        op2_bank_addr     := io.cmdReq.bits.cmd.op2_bank_addr
        wr_bank           := io.cmdReq.bits.cmd.wr_bank
        wr_bank_addr      := io.cmdReq.bits.cmd.wr_bank_addr
        state             := busy
        io.is_matmul_ws   := true.B
      }
    }
    is(busy) {
      iteration_counter := iteration_counter + 1.U
      when(iteration_counter === iteration - 1.U) {
        iteration_counter := 0.U
        state := idle
    }
    }
  }
  // Generate ID_LU request
  io.id_lu_o.valid              := state === busy
  io.id_lu_o.bits.op1_bank      := op1_bank
  io.id_lu_o.bits.op1_bank_addr := op1_bank_addr + b.veclane.U - iteration_counter - 1.U
  io.id_lu_o.bits.op2_bank      := op2_bank
  io.id_lu_o.bits.op2_bank_addr := op2_bank_addr + iteration_counter
  io.id_lu_o.bits.wr_bank       := wr_bank
  io.id_lu_o.bits.wr_bank_addr  := wr_bank_addr + iteration_counter
  io.id_lu_o.bits.opcode        := 1.U
  io.id_lu_o.bits.iter          := iteration
  io.id_lu_o.bits.thread_id     := iteration_counter
  io.id_lu_o.bits.rob_id        := rob_id_reg

  io.cmdReq.ready := io.id_lu_o.ready

  // Instruction completion signal


  // Delay complete signal by 10 cycles
  // val complete_delay = RegInit(VecInit(Seq.fill(10)(false.B)))
  // complete_delay(0) := complete
  // for (i <- 1 until 10) {
  //   complete_delay(i) := complete_delay(i-1)
  // }
  // val complete_10clk = complete_delay(9)


}
