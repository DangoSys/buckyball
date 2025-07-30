package dialect.bbfp

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import dialect.bbfp._
import buckyball.frontend.rs.{ReservationStationIssue, ReservationStationComplete, BuckyBallCmd}
import buckyball.mem.{SramReadIO, SramWriteIO}
import buckyball.BuckyBallConfig
import org.yaml.snakeyaml.events.Event.ID

class BBFP_Control(implicit bbconfig: BuckyBallConfig, p: Parameters) extends Module {
    val rob_id_width = log2Up(bbconfig.rob_entries)
    val spad_w = bbconfig.veclane * bbconfig.inputType.getWidth
  
    val io = IO(new Bundle {
        val cmdReq = Flipped(Decoupled(new ReservationStationIssue(new BuckyBallCmd, rob_id_width)))
        val cmdResp = Decoupled(new ReservationStationComplete(rob_id_width))
        val is_matmul_ws = Input(Bool())
        // 连接到Scratchpad的SRAM读写接口
        val sramRead = Vec(bbconfig.sp_banks, Flipped(new SramReadIO(bbconfig.spad_bank_entries, spad_w)))
        val sramWrite = Vec(bbconfig.sp_banks, Flipped(new SramWriteIO(bbconfig.spad_bank_entries, spad_w, bbconfig.spad_mask_len)))

         // 连接到Accumulator的读写接口
        val accRead = Vec(bbconfig.acc_banks, Flipped(new SramReadIO(bbconfig.acc_bank_entries, bbconfig.acc_w)))
        val accWrite = Vec(bbconfig.acc_banks, Flipped(new SramWriteIO(bbconfig.acc_bank_entries, bbconfig.acc_w, bbconfig.acc_mask_len)))
  })
// -----------------------------------------------------------------------------
// BBFP_ID
// -----------------------------------------------------------------------------
    val BBFP_ID = Module(new BBFP_ID)
    BBFP_ID.io.cmdReq <> io.cmdReq
    io.cmdResp <> BBFP_ID.io.cmdResp
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
    for (i <- 0 until bbconfig.sp_banks) {
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
    for (i <- 0 until bbconfig.sp_banks) {
        BBFP_EX.io.sramReadResp(i) <> io.sramRead(i).resp
        io.sramWrite(i) <> BBFP_EX.io.sramWrite(i)
    }
    BBFP_EX.io.is_matmul_ws := io.is_matmul_ws
    for (i <- 0 until bbconfig.acc_banks) {
        io.accWrite(i) <> BBFP_EX.io.accWrite(i)
        io.accRead(i) := DontCare
    }

}