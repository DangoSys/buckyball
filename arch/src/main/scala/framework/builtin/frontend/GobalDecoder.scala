package framework.builtin.frontend

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import freechips.rocketchip.tile._
import framework.builtin.memdomain.DISA._
import framework.builtin.frontend.GISA._
class BuckyballRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}



class PostGDCmd(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  // Instruction type determination
  // Ball instruction (excluding FENCE)
  val is_ball       = Bool()
  // Memory instruction (load/store)
  val is_mem        = Bool()
  // Fence instruction
  val is_fence      = Bool()

  // Raw instruction information, passed to corresponding domain decoder
  val raw_cmd       = new RoCCCommand
}

class GlobalDecoder(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommand
    }))
    val id_o = Decoupled(new PostGDCmd)
  })

  // If reservation station is blocked, id_i is also blocked
  io.id_i.ready := io.id_o.ready

  val func7 = io.id_i.bits.cmd.inst.funct

  // Instruction type determination: distinguish Ball, Mem, Fence instructions
  val is_mem_instr   = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
  val is_fence_instr = (func7 === FENCE_BITPAT)
  val is_ball_instr  = !is_mem_instr && !is_fence_instr

  // Output control
  io.id_o.valid          := io.id_i.valid

  io.id_o.bits.is_ball   := is_ball_instr
  io.id_o.bits.is_mem    := is_mem_instr
  io.id_o.bits.is_fence  := is_fence_instr
  io.id_o.bits.raw_cmd   := io.id_i.bits.cmd
}
