package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import examples.toy.balldomain.{ExReservationStationIssue, ExReservationStationComplete, ExBuckyBallCmd}
import framework.builtin.mem.{SramReadIO, SramWriteIO}
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class BBFP_ID(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Module {
    val rob_id_width = log2Up(bbconfig.rob_entries)
    val spad_w = bbconfig.veclane * bbconfig.inputType.getWidth

    val io = IO(new Bundle{
        val cmdReq = Flipped(Decoupled(new ExReservationStationIssue))
        val cmdResp = Decoupled(new ExReservationStationComplete)
        val is_matmul_ws  = Output(Bool())
        val id_lu_o = Decoupled(new id_lu_req)
    })
    
    val busy :: idle :: Nil = Enum(2)
    //寄存器定义
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
            when(io.cmdReq.valid && io.cmdReq.bits.cmd.ex_decode_cmd.is_bbfp) {
                iteration := io.cmdReq.bits.cmd.ex_decode_cmd.iter
                iteration_counter := 0.U
                is_matmul_ws := false.B
                rob_id_reg := io.cmdReq.bits.rob_id
                op1_bank := io.cmdReq.bits.cmd.ex_decode_cmd.op1_bank
                op1_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.op1_bank_addr
                op2_bank := io.cmdReq.bits.cmd.ex_decode_cmd.op2_bank
                op2_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.op2_bank_addr
                wr_bank := io.cmdReq.bits.cmd.ex_decode_cmd.wr_bank
                wr_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.wr_bank_addr
                state := busy
                io.is_matmul_ws := false.B
            }
            when(io.cmdReq.valid && io.cmdReq.bits.cmd.ex_decode_cmd.is_matmul_ws){
                iteration := io.cmdReq.bits.cmd.ex_decode_cmd.iter 
                iteration_counter := 0.U
                rob_id_reg := io.cmdReq.bits.rob_id
                op1_bank := io.cmdReq.bits.cmd.ex_decode_cmd.op1_bank
                op1_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.op1_bank_addr
                op2_bank := io.cmdReq.bits.cmd.ex_decode_cmd.op2_bank
                op2_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.op2_bank_addr
                wr_bank := io.cmdReq.bits.cmd.ex_decode_cmd.wr_bank
                wr_bank_addr := io.cmdReq.bits.cmd.ex_decode_cmd.wr_bank_addr
                state := busy
                io.is_matmul_ws := true.B
                is_matmul_ws := true.B
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
    //生成ID_LU请求
    io.id_lu_o.valid := state === busy
    io.id_lu_o.bits.op1_bank := op1_bank
    io.id_lu_o.bits.op1_bank_addr := op1_bank_addr + iteration_counter
    io.id_lu_o.bits.op2_bank := op2_bank
    io.id_lu_o.bits.op2_bank_addr := op2_bank_addr + iteration_counter
    io.id_lu_o.bits.wr_bank := wr_bank
    io.id_lu_o.bits.wr_bank_addr := wr_bank_addr + iteration_counter
    io.id_lu_o.bits.opcode := 1.U
    io.id_lu_o.bits.iter := iteration
    io.id_lu_o.bits.thread_id := iteration_counter

    io.cmdReq.ready := io.id_lu_o.ready

    //指令完成信号
    val complete = (iteration_counter === iteration - 1.U) && (state === busy) 
    
    // 将complete信号打10拍
    // val complete_delay = RegInit(VecInit(Seq.fill(10)(false.B)))
    // complete_delay(0) := complete
    // for (i <- 1 until 10) {
    //     complete_delay(i) := complete_delay(i-1)
    // }
    // val complete_10clk = complete_delay(9)
    
    io.cmdResp.bits.rob_id := rob_id_reg
    io.cmdResp.valid := complete

}