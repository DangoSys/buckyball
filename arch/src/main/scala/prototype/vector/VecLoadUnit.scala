package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import examples.toy.balldomain.BallDomainParam

class ctrl_ld_req(parameter: BallDomainParam) extends Bundle {
  val op1_bank      = UInt(log2Up(parameter.numBanks).W)
  val op1_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val op2_bank      = UInt(log2Up(parameter.numBanks).W)
  val op2_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val iter          = UInt(10.W)
  val mode          = UInt(1.W)
}

@instantiable
class VecLoadUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum     = 16
  val inputWidth   = 8
  val bankWidth    = parameter.bankWidth
  val rob_id_width = log2Up(parameter.rob_entries)

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(parameter.numBanks, Decoupled(new SramReadReq(parameter.bankEntries)))
    val bankReadResp = Vec(parameter.numBanks, Flipped(Decoupled(new SramReadResp(bankWidth))))
    val ctrl_ld_i    = Flipped(Decoupled(new ctrl_ld_req(parameter)))
    val ld_ex_o      = Decoupled(new ld_ex_req(parameter))
  })

  val op1_bank     = RegInit(0.U(log2Up(parameter.numBanks).W))
  val op2_bank     = RegInit(0.U(log2Up(parameter.numBanks).W))
  val op1_addr     = RegInit(0.U(log2Up(parameter.bankEntries).W))
  val op2_addr     = RegInit(0.U(log2Up(parameter.bankEntries).W))
  val iter         = RegInit(0.U(10.W))
  val iter_counter = RegInit(0.U(10.W))
  val mode         = RegInit(0.U(1.W))

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  // Output register to break combinational logic loop
  val ld_ex_valid_reg = RegInit(false.B)
  val ld_ex_op1_reg   = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_op2_reg   = Reg(Vec(InputNum, UInt(inputWidth.W)))
  val ld_ex_iter_reg  = RegInit(0.U(10.W))

  // Default assignment for each bank read request
  for (i <- 0 until parameter.numBanks) {
    io.bankReadReq(i).valid        := false.B
    io.bankReadReq(i).bits.fromDMA := false.B
    io.bankReadReq(i).bits.addr    := 0.U
  }

  io.ctrl_ld_i.ready := state === idle

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------

  when(io.ctrl_ld_i.fire) {
    op1_bank     := io.ctrl_ld_i.bits.op1_bank
    op2_bank     := io.ctrl_ld_i.bits.op2_bank
    op1_addr     := io.ctrl_ld_i.bits.op1_bank_addr
    op2_addr     := io.ctrl_ld_i.bits.op2_bank_addr
    iter         := io.ctrl_ld_i.bits.iter
    iter_counter := 0.U
    state        := busy
    mode         := io.ctrl_ld_i.bits.mode
    assert(io.ctrl_ld_i.bits.iter > 0.U, "iter should be greater than 0")
  }
  io.bankReadResp.foreach { resp =>
    resp.ready := state === busy
  }
  when(mode === 0.U) {
// -----------------------------------------------------------------------------
// Send SRAM read request (only when output register is idle)
// -----------------------------------------------------------------------------
    when(state === busy && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      io.bankReadReq(op1_bank).valid        := iter_counter < iter
      io.bankReadReq(op1_bank).bits.fromDMA := false.B
      io.bankReadReq(op1_bank).bits.addr    := op1_addr + iter_counter

      io.bankReadReq(op2_bank).valid        := iter_counter < iter
      io.bankReadReq(op2_bank).bits.fromDMA := false.B
      io.bankReadReq(op2_bank).bits.addr    := op2_addr + iter_counter
      iter_counter                          := iter_counter + 1.U
    }

// -----------------------------------------------------------------------------
// SRAM returns data and passes to EX unit (use register to break combinational logic loop)
// -----------------------------------------------------------------------------
    // ready signal for bankReadResp: can receive when there's no pending data or downstream has received
    /* io.bankReadResp.foreach { resp => resp.ready := !ld_ex_valid_reg || io.ld_ex_o.ready } */
    // Receive SRAM data and cache to register
    when(io.bankReadResp(op1_bank).valid && io.bankReadResp(op2_bank).valid &&
      (!ld_ex_valid_reg || io.ld_ex_o.ready) && (state === busy)) {
      ld_ex_valid_reg := true.B
      ld_ex_op1_reg   := io.bankReadResp(op1_bank).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      ld_ex_op2_reg   := io.bankReadResp(op2_bank).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      ld_ex_iter_reg  := iter_counter
    }.elsewhen(io.ld_ex_o.ready) {
      ld_ex_valid_reg := false.B
    }

    // Output comes from register
    io.ld_ex_o.valid     := ld_ex_valid_reg
    io.ld_ex_o.bits.op1  := ld_ex_op1_reg
    io.ld_ex_o.bits.op2  := ld_ex_op2_reg
    io.ld_ex_o.bits.iter := ld_ex_iter_reg

// -----------------------------------------------------------------------------
// Reset iter_counter and return to idle state
// -----------------------------------------------------------------------------

    when(state === busy && iter_counter === iter && (!ld_ex_valid_reg || io.ld_ex_o.ready)) {
      state        := idle
      iter_counter := 0.U
    }
  }.otherwise {
    // Default assignment
    io.ld_ex_o.valid     := false.B
    io.ld_ex_o.bits.op1  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    io.ld_ex_o.bits.op2  := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    io.ld_ex_o.bits.iter := 0.U
    when(state === busy && io.bankReadResp(0).valid) {
      iter_counter                   := iter_counter + 1.U
      ld_ex_op1_reg                  := io.bankReadResp(0).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.bankReadReq(1).valid        := true.B
      io.bankReadReq(1).bits.addr    := op2_addr + iter_counter
      io.bankReadReq(1).bits.fromDMA := false.B
    }
    when(state === busy && io.bankReadResp(1).valid && RegNext(io.bankReadResp(0).valid)) {
      io.ld_ex_o.valid     := true.B
      io.ld_ex_o.bits.op1  := ld_ex_op1_reg
      io.ld_ex_o.bits.op2  := io.bankReadResp(1).bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
      io.ld_ex_o.bits.iter := iter_counter - 1.U
    }
    when(state === busy && iter_counter === iter) {
      state        := idle
      iter_counter := 0.U
    }
  }

}
