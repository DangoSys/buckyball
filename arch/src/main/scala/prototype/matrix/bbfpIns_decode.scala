package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam

@instantiable
class BBFP_ID(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum     = 16
  val inputWidth   = 8
  val rob_id_width = log2Up(parameter.rob_entries)
  val bankWidth    = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(parameter)))
    val is_matmul_ws = Output(Bool())
    val id_lu_o      = Decoupled(new id_lu_req(parameter))
  })

  val idle :: busy :: Nil = Enum(2)
  // Register definitions
  val state               = RegInit(idle)
  val rob_id_reg          = RegInit(0.U(rob_id_width.W))
  val iteration_counter   = RegInit(0.U(10.W))
  val iteration           = RegInit(0.U(10.W))
  val op1_bank            = RegInit(0.U(log2Up(parameter.numBanks).W))
  val op1_bank_addr       = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val op2_bank_addr       = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val op2_bank            = RegInit(0.U(log2Up(parameter.numBanks).W))
  val wr_bank             = RegInit(0.U(log2Up(parameter.numBanks).W))
  val wr_bank_addr        = RegInit(0.U(12.W)) // New ISA: always 0, but keep for compatibility
  val is_matmul_ws        = RegInit(false.B)
  io.is_matmul_ws := false.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.valid && io.cmdReq.bits.cmd.bid === 1.U) {
        iteration         := io.cmdReq.bits.cmd.iter
        iteration_counter := 0.U
        is_matmul_ws      := false.B
        rob_id_reg        := io.cmdReq.bits.rob_id
        op1_bank          := io.cmdReq.bits.cmd.op1_bank
        op1_bank_addr     := 0.U // New ISA: all operations start from row 0
        op2_bank          := io.cmdReq.bits.cmd.op2_bank
        op2_bank_addr     := 0.U // New ISA: all operations start from row 0
        wr_bank           := io.cmdReq.bits.cmd.wr_bank
        wr_bank_addr      := 0.U // New ISA: all operations start from row 0
        state             := busy
        io.is_matmul_ws   := false.B
      }
      when(io.cmdReq.valid && io.cmdReq.bits.cmd.special(0)) {
        iteration         := io.cmdReq.bits.cmd.iter
        iteration_counter := 0.U
        is_matmul_ws      := true.B
        rob_id_reg        := io.cmdReq.bits.rob_id
        op1_bank          := io.cmdReq.bits.cmd.op1_bank
        op1_bank_addr     := 0.U // New ISA: all operations start from row 0
        op2_bank          := io.cmdReq.bits.cmd.op2_bank
        op2_bank_addr     := 0.U // New ISA: all operations start from row 0
        wr_bank           := io.cmdReq.bits.cmd.wr_bank
        wr_bank_addr      := 0.U // New ISA: all operations start from row 0
        state             := busy
        io.is_matmul_ws   := true.B
      }
    }
    is(busy) {
      iteration_counter := iteration_counter + 1.U
      when(iteration_counter === iteration - 1.U) {
        iteration_counter := 0.U
        state             := idle
      }
    }
  }
  // Generate ID_LU request
  io.id_lu_o.valid              := state === busy
  io.id_lu_o.bits.op1_bank      := op1_bank
  io.id_lu_o.bits.op1_bank_addr := op1_bank_addr + InputNum.U - iteration_counter - 1.U
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
