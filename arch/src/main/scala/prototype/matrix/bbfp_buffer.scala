package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class id_lu_req(implicit b: CustomBuckyBallConfig) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks).W)
  val op1_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val op2_bank      = UInt(log2Up(b.sp_banks).W)
  val op2_bank_addr = UInt(log2Up(b.spad_bank_entries).W)
  val wr_bank       = UInt(log2Up(b.sp_banks).W)
  val wr_bank_addr  = UInt(log2Up(b.spad_bank_entries).W)
  val opcode        = UInt(3.W)
  val iter          = UInt(10.W) 
  val thread_id     = UInt(10.W)
  val rob_id        = UInt(log2Up(b.rob_entries).W)
}

class lu_ex_req(implicit b: CustomBuckyBallConfig) extends Bundle {
  val op1_bank      = UInt(log2Up(b.sp_banks).W)
  val op2_bank      = UInt(log2Up(b.sp_banks).W)
  val wr_bank       = UInt(log2Up(b.sp_banks).W)
  val wr_bank_addr  = UInt(log2Up(b.spad_bank_entries).W)
  val opcode				= UInt(3.W)
  val iter  				= UInt(10.W)
  val thread_id   	= UInt(10.W)
  val rob_id        = UInt(log2Up(b.rob_entries).W)
}

class ID_LU(implicit b: CustomBuckyBallConfig) extends Module{
  val io = IO(new Bundle {
    val id_lu_i = Flipped(Decoupled(new id_lu_req))
    val ld_lu_o = Decoupled(new id_lu_req)
  })
    // 1-cycle delay register
  val delayed_req = RegEnable(io.id_lu_i.bits, io.id_lu_i.fire)
  val delayed_valid = RegNext(io.id_lu_i.valid, false.B)

  // Output connection
  io.ld_lu_o.bits := delayed_req
  io.ld_lu_o.valid := delayed_valid

  // Backpressure: input is ready if output is ready (since we have a 1-slot buffer)
  io.id_lu_i.ready := io.ld_lu_o.ready
}

class LU_EX(implicit b: CustomBuckyBallConfig) extends Module{
  val io = IO(new Bundle {
    val lu_ex_i = Flipped(Decoupled(new lu_ex_req))
    val lu_ex_o = Decoupled(new lu_ex_req)
  })
    // 1-cycle delay register
  val delayed_req = RegEnable(io.lu_ex_i.bits, io.lu_ex_i.fire)
  val delayed_valid = RegNext(io.lu_ex_i.valid, false.B)

  // Output connection
  io.lu_ex_o.bits := delayed_req
  io.lu_ex_o.valid := delayed_valid

  // Backpressure: input is ready if output is ready (since we have a 1-slot buffer)
  io.lu_ex_i.ready := io.lu_ex_o.ready
}