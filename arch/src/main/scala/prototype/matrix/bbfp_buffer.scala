package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import examples.toy.balldomain.BallDomainParam

class id_lu_req(parameter: BallDomainParam) extends Bundle {
  val op1_bank      = UInt(log2Up(parameter.numBanks).W)
  val op1_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val op2_bank      = UInt(log2Up(parameter.numBanks).W)
  val op2_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val wr_bank       = UInt(log2Up(parameter.numBanks).W)
  val wr_bank_addr  = UInt(log2Up(parameter.bankEntries).W)
  val opcode        = UInt(3.W)
  val iter          = UInt(10.W)
  val thread_id     = UInt(10.W)
  val rob_id        = UInt(log2Up(parameter.rob_entries).W)
}

class lu_ex_req(parameter: BallDomainParam) extends Bundle {
  val op1_bank     = UInt(log2Up(parameter.numBanks).W)
  val op2_bank     = UInt(log2Up(parameter.numBanks).W)
  val wr_bank      = UInt(log2Up(parameter.numBanks).W)
  val wr_bank_addr = UInt(log2Up(parameter.bankEntries).W)
  val opcode       = UInt(3.W)
  val iter         = UInt(10.W)
  val thread_id    = UInt(10.W)
  val rob_id       = UInt(log2Up(parameter.rob_entries).W)
}

@instantiable
class ID_LU(val parameter: BallDomainParam) extends Module with SerializableModule[BallDomainParam] {

  @public
  val io = IO(new Bundle {
    val id_lu_i = Flipped(Decoupled(new id_lu_req(parameter)))
    val ld_lu_o = Decoupled(new id_lu_req(parameter))
  })

  // 1-cycle delay register
  val delayed_req   = RegEnable(io.id_lu_i.bits, io.id_lu_i.fire)
  val delayed_valid = RegNext(io.id_lu_i.valid, false.B)

  // Output connection
  io.ld_lu_o.bits  := delayed_req
  io.ld_lu_o.valid := delayed_valid

  // Backpressure: input is ready if output is ready (since we have a 1-slot buffer)
  io.id_lu_i.ready := io.ld_lu_o.ready
}

@instantiable
class LU_EX(val parameter: BallDomainParam) extends Module with SerializableModule[BallDomainParam] {

  @public
  val io = IO(new Bundle {
    val lu_ex_i = Flipped(Decoupled(new lu_ex_req(parameter)))
    val lu_ex_o = Decoupled(new lu_ex_req(parameter))
  })

  // 1-cycle delay register
  val delayed_req   = RegEnable(io.lu_ex_i.bits, io.lu_ex_i.fire)
  val delayed_valid = RegNext(io.lu_ex_i.valid, false.B)

  // Output connection
  io.lu_ex_o.bits  := delayed_req
  io.lu_ex_o.valid := delayed_valid

  // Backpressure: input is ready if output is ready (since we have a 1-slot buffer)
  io.lu_ex_i.ready := io.lu_ex_o.ready
}
