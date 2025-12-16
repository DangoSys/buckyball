package framework.gpdomain.sequencer.decoder

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._


class BuckyballRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}

object DISA {
  // RVV Instruction Opcodes
  val RVV_OPCODE_V   = "b1010111".U // 0x57: OP-V (vector compute)
  val RVV_OPCODE_VL  = "b0000111".U // 0x07: LOAD-FP (vector load)
  val RVV_OPCODE_VS  = "b0100111".U // 0x27: STORE-FP (vector store)
}
