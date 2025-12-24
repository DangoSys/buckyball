package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.memdomain.backend.banks.SramReadReq
import examples.toy.balldomain.BallDomainParam

@instantiable
class BBFP_LoadUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum     = 16
  val inputWidth   = 8
  val rob_id_width = log2Up(parameter.rob_entries)
  val bankWidth    = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    val sramReadReq = Vec(parameter.numBanks, Decoupled(new SramReadReq(parameter.bankEntries)))
    val id_lu_i     = Flipped(Decoupled(new id_lu_req(parameter)))
    val lu_ex_o     = Decoupled(new lu_ex_req(parameter))
  })

  val op1_bank      = io.id_lu_i.bits.op1_bank
  val op1_bank_addr = io.id_lu_i.bits.op1_bank_addr
  val op2_bank      = io.id_lu_i.bits.op2_bank
  val op2_bank_addr = io.id_lu_i.bits.op2_bank_addr
  val wr_bank       = io.id_lu_i.bits.wr_bank
  val wr_bank_addr  = io.id_lu_i.bits.wr_bank_addr

  // Default assignment for each bank read request
  for (i <- 0 until parameter.numBanks) {
    io.sramReadReq(i).valid        := false.B
    io.sramReadReq(i).bits.fromDMA := false.B
    io.sramReadReq(i).bits.addr    := 0.U
  }

  // Generate SRAM read request based on ID_LU input
  when(io.id_lu_i.valid) {
    io.sramReadReq(op1_bank).valid        := true.B
    io.sramReadReq(op1_bank).bits.fromDMA := false.B
    io.sramReadReq(op1_bank).bits.addr    := op1_bank_addr

    io.sramReadReq(op2_bank).valid        := true.B
    io.sramReadReq(op2_bank).bits.fromDMA := false.B
    io.sramReadReq(op2_bank).bits.addr    := op2_bank_addr
  }

  // Generate LU_EX request
  io.lu_ex_o.valid             := io.id_lu_i.valid
  io.lu_ex_o.bits.op1_bank     := op1_bank
  io.lu_ex_o.bits.op2_bank     := op2_bank
  io.lu_ex_o.bits.wr_bank      := wr_bank
  io.lu_ex_o.bits.wr_bank_addr := wr_bank_addr
  io.lu_ex_o.bits.opcode       := io.id_lu_i.bits.opcode
  io.lu_ex_o.bits.iter         := io.id_lu_i.bits.iter
  io.lu_ex_o.bits.thread_id    := io.id_lu_i.bits.thread_id
  io.lu_ex_o.bits.rob_id       := io.id_lu_i.bits.rob_id

  io.id_lu_i.ready := io.lu_ex_o.ready

}
