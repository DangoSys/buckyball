package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import org.yaml.snakeyaml.events.Event.ID
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}

class BBFP_Control(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)
    val is_matmul_ws = Input(Bool())
    // 连接到Scratchpad的SRAM读写接口
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))

     // 连接到Accumulator的读写接口
    // val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })
// -----------------------------------------------------------------------------
// BBFP_ID
// -----------------------------------------------------------------------------
  val BBFP_ID = Module(new BBFP_ID)
  BBFP_ID.io.cmdReq <> io.cmdReq
// -----------------------------------------------------------------------------
// ID_LU
// -----------------------------------------------------------------------------
  val ID_LU = Module(new ID_LU)
  ID_LU.io.id_lu_i <> BBFP_ID.io.id_lu_o

// -----------------------------------------------------------------------------
// BBFP_LoadUnit
// -----------------------------------------------------------------------------
  val BBFP_LoadUnit = Module(new BBFP_LoadUnit)
  BBFP_LoadUnit.io.id_lu_i <> ID_LU.io.ld_lu_o
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req <> BBFP_LoadUnit.io.sramReadReq(i)
  }
// -----------------------------------------------------------------------------
// LU_EX
// -----------------------------------------------------------------------------
  val LU_EX = Module(new LU_EX)
  LU_EX.io.lu_ex_i <> BBFP_LoadUnit.io.lu_ex_o

// -----------------------------------------------------------------------------
// BBFP_EX
// -----------------------------------------------------------------------------
  val BBFP_EX = Module(new BBFP_EX)
  BBFP_EX.io.lu_ex_i <> LU_EX.io.lu_ex_o
  for (i <- 0 until b.sp_banks) {
    BBFP_EX.io.sramReadResp(i) <> io.sramRead(i).resp
    io.sramWrite(i) <> BBFP_EX.io.sramWrite(i)
  }
  BBFP_EX.io.is_matmul_ws := io.is_matmul_ws
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i) <> BBFP_EX.io.accWrite(i)
    // io.accRead(i) := DontCare
  }
  io.cmdResp <> BBFP_EX.io.cmdResp

}
