package framework.builtin.frontend

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import freechips.rocketchip.tile._
import framework.builtin.memdomain.DISA._
import framework.builtin.frontend.GISA._
import framework.rocket.RoCCCommandBB

class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommandBB
}



class PostGDCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // 指令类型判断
  val is_ball       = Bool()  // Ball指令(不包括FENCE)
  val is_mem        = Bool()  // 内存指令(load/store)
  val is_fence      = Bool()  // Fence指令

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

  // 指令类型判断：区分Ball、Mem、Fence指令
  val is_mem_instr   = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
  val is_fence_instr = (func7 === FENCE_BITPAT)
  val is_ball_instr  = !is_mem_instr && !is_fence_instr

  // 输出控制
  io.id_o.valid          := io.id_i.valid

  io.id_o.bits.is_ball   := is_ball_instr
  io.id_o.bits.is_mem    := is_mem_instr
  io.id_o.bits.is_fence  := is_fence_instr
  io.id_o.bits.raw_cmd   := io.id_i.bits.cmd
}
