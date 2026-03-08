package framework.frontend.decoder

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import freechips.rocketchip.tile._

import framework.frontend.decoder.GISA._
import framework.memdomain.frontend.cmd_channel.decoder.DISA._
import framework.gpdomain.sequencer.decoder.DISA._
import framework.frontend.scoreboard.BankAccessInfo

import framework.core.bbtile.RoCCCommandBB

class BuckyballRawCmd(val b: GlobalConfig) extends Bundle {
  val cmd = new RoCCCommandBB(b.core.xLen)
}

class PostGDCmd(val b: GlobalConfig) extends Bundle {
  val domain_id  = UInt(4.W)
  val cmd        = new RoCCCommandBB(b.core.xLen)
  val bankAccess = new BankAccessInfo(log2Up(b.memDomain.bankNum))
  val isFence    = Bool()
}

@instantiable
class GlobalDecoder(val b: GlobalConfig) extends Module {

  val bankIdLen = b.frontend.bank_id_len

  @public
  val io = IO(new Bundle {

    val id_i = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.core.xLen)
    }))

    val id_o = Decoupled(new PostGDCmd(b))
  })

  // If reservation station is blocked, id_i is also blocked
  io.id_i.ready := io.id_o.ready

  val func7  = io.id_i.bits.cmd.funct
  val opcode = io.id_i.bits.cmd.opcode
  val rs1    = io.id_i.bits.cmd.rs1Data

  // Instruction type determination: distinguish Ball, Mem, Fence, GP (RVV) instructions
  val is_mem_inst = (func7 === MVIN_BITPAT) ||
    (func7 === MVOUT_BITPAT) ||
    (func7 === MSET_BITPAT)

  val is_frontend_inst = func7 === FENCE_BITPAT

  // RVV instructions: opcode 0x57 (vector compute), 0x07 (vector load), 0x27 (vector store)
  val is_gp_inst = (opcode === RVV_OPCODE_V) ||
    (opcode === RVV_OPCODE_VL) ||
    (opcode === RVV_OPCODE_VS)

  val is_ball_inst = !is_mem_inst && !is_frontend_inst && !is_gp_inst

  // Encode domain ID
  val domain_id = MuxCase(
    DomainId.BALL,
    Seq(
      is_frontend_inst -> DomainId.FRONTEND,
      is_mem_inst      -> DomainId.MEM,
      is_gp_inst       -> DomainId.GP,
      is_ball_inst     -> DomainId.BALL
    )
  )

  // -------------------------------------------------------------------------
  // Bank access info extraction — read valid flags from rs1[45:47]
  //
  // Unified rs1 layout (defined in isa.h):
  //   rs1[14:0]  = bank_0  (rd_bank_0 or wr_bank for MVIN/MSET)
  //   rs1[29:15] = bank_1  (rd_bank_1, dual-operand only)
  //   rs1[44:30] = bank_2  (wr_bank for Ball instructions)
  //   rs1[45]    = rd_bank_0_valid flag (BB_RD0)
  //   rs1[46]    = rd_bank_1_valid flag (BB_RD1)
  //   rs1[47]    = wr_bank_valid flag (BB_WR)
  //   rs1[63:48] = iter (16-bit)
  // -------------------------------------------------------------------------
  val bankAccess = Wire(new BankAccessInfo(bankIdLen))

  bankAccess.rd_bank_0_valid := rs1(45)
  bankAccess.rd_bank_0_id    := rs1(bankIdLen - 1, 0)
  bankAccess.rd_bank_1_valid := rs1(46)
  bankAccess.rd_bank_1_id    := rs1(bankIdLen + 14, 15)
  bankAccess.wr_bank_valid   := rs1(47)
  // For Mem instructions (MVIN/MSET), wr_bank is bank_0 (rs1[14:0])
  // For Ball instructions, wr_bank is bank_2 (rs1[44:30])
  bankAccess.wr_bank_id      := Mux(is_mem_inst, rs1(bankIdLen - 1, 0), rs1(bankIdLen + 29, 30))

  // Output control
  io.id_o.valid           := io.id_i.valid
  io.id_o.bits.domain_id  := domain_id
  io.id_o.bits.cmd        := io.id_i.bits.cmd
  io.id_o.bits.bankAccess := bankAccess
  io.id_o.bits.isFence    := is_frontend_inst
}
