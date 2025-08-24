package framework.builtin.frontend

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import freechips.rocketchip.tile._  
import framework.builtin.memdomain.DISA._
import framework.rocket.RoCCCommandBB

class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommandBB
}



class PostGDCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // 指令类型判断
  val is_ball       = Bool()  // Ball指令(包括FENCE)
  val is_mem        = Bool()  // 内存指令(load/store)
  
  // 原始指令信息，传递给对应的domain decoder
  val raw_cmd       = new RoCCCommandBB
}

class GlobalDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB
    }))
    val id_o = Decoupled(new PostGDCmd)
  })

  io.id_i.ready := io.id_o.ready // 如果保留站阻塞了，id_i也阻塞

  val func7 = io.id_i.bits.cmd.inst.funct

  // 指令类型判断：只区分EX指令和Mem指令
  val is_mem_instr = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
  val is_ball_instr = !is_mem_instr  // Ball指令包含所有非内存指令(包括FENCE)

  // 输出控制
  io.id_o.valid        := io.id_i.valid
  io.id_o.bits.is_ball   := is_ball_instr
  io.id_o.bits.is_mem  := is_mem_instr
  io.id_o.bits.raw_cmd := io.id_i.bits.cmd
}


