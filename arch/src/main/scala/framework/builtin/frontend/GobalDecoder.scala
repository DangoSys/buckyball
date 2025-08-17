package framework.builtin.frontend

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import freechips.rocketchip.tile._  
import framework.builtin.memdomain.DISA._


class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}



class PostDecodeCmd(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  // 指令类型判断
  val is_ex         = Bool()  // EX指令(包括FENCE)
  val is_mem        = Bool()  // 内存指令(load/store)
  
  // 原始指令信息，传递给对应的domain decoder
  val raw_cmd       = new RoCCCommand
}

class GlobalDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommand
    }))
    val id_rs = Decoupled(new PostDecodeCmd)
  })

  io.id_i.ready := io.id_rs.ready // 如果保留站阻塞了，id_i也阻塞

  val func7 = io.id_i.bits.cmd.inst.funct

  // 指令类型判断：只区分EX指令和Mem指令
  val is_mem_instr = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
  val is_ex_instr = !is_mem_instr  // EX指令包含所有非内存指令(包括FENCE)

  // 输出控制
  io.id_rs.valid := io.id_i.valid
  io.id_rs.bits.is_ex := is_ex_instr
  io.id_rs.bits.is_mem := is_mem_instr
  io.id_rs.bits.raw_cmd := io.id_i.bits.cmd
}


