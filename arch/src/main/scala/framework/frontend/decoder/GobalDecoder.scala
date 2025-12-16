package framework.frontend.decoder

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import freechips.rocketchip.tile._

import framework.frontend.decoder.GISA._
import framework.memdomain.DISA._
import framework.gpdomain.sequencer.decoder.DISA._

class BuckyballRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}

class PostGDCmd(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val domain_id     = UInt(4.W)
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
  val opcode = io.id_i.bits.cmd.inst.opcode

  // Instruction type determination: distinguish Ball, Mem, Fence, GP (RVV) instructions
  val is_mem_inst      = (func7 === MVIN_BITPAT) || (func7 === MVOUT_BITPAT)
  val is_frontend_inst = (func7 === FENCE_BITPAT)
  // RVV instructions: opcode 0x57 (vector compute), 0x07 (vector load), 0x27 (vector store)
  val is_gp_inst       = (opcode === RVV_OPCODE_V) || (opcode === RVV_OPCODE_VL) || (opcode === RVV_OPCODE_VS)
  val is_ball_inst     = !is_mem_inst && !is_frontend_inst && !is_gp_inst

  // Encode domain ID
  val domain_id = MuxCase(DomainId.BALL, Seq(
    is_frontend_inst -> DomainId.FRONTEND,
    is_mem_inst      -> DomainId.MEM,
    is_gp_inst       -> DomainId.GP,
    is_ball_inst     -> DomainId.BALL
  ))

  // Output control
  io.id_o.valid          := io.id_i.valid
  io.id_o.bits.domain_id := domain_id
  io.id_o.bits.raw_cmd   := io.id_i.bits.cmd
}
