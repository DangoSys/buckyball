// Copied from riscv-boom src/main/scala/v4/exu/decode.scala and modified for Buckyball.
// Copyright (c) The Regents of the University of California (Regents).
// All Rights Reserved. See LICENSE and LICENSE.SiFive in riscv-boom for license details.

package examples.core.boom.native

import chisel3._
import chisel3.util._

import freechips.rocketchip.tile.FPConstants
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.rocket.Instructions._
import freechips.rocketchip.rocket.Instructions32
import freechips.rocketchip.rocket.CustomInstructions._
import freechips.rocketchip.rocket.RVCExpander
import freechips.rocketchip.rocket.ALU._
import freechips.rocketchip.rocket.{CSR, Causes, DecodeLogic}
import freechips.rocketchip.util._

import boom.v4.common._
import boom.v4.util._
import boom.v4.exu._

/**
 * Decode unit that takes in a single instruction and generates a MicroOp.
 */
class DecodeUnit(implicit p: Parameters)
    extends BoomModule
    with freechips.rocketchip.rocket.constants.MemoryOpConstants
    with freechips.rocketchip.rocket.constants.ScalarOpConstants {
  val io = IO(new DecodeUnitIo)

  val uop = Wire(new MicroOp())
  uop := io.enq.uop

  val decode_table =
    DecodeTables.X_table ++
      DecodeTables.F_table ++
      DecodeTables.FDivSqrt_table ++
      DecodeTables.X64_table ++
      DecodeTables.B_table ++
      (if (usingRoCC) DecodeTables.RoCC_table else Nil)

  val inst = uop.inst
  val LDST = inst(RD_MSB, RD_LSB)
  val LRS1 = inst(RS1_MSB, RS1_LSB)
  val LRS2 = inst(RS2_MSB, RS2_LSB)
  val LRS3 = inst(RS3_MSB, RS3_LSB)

  val cs = Wire(new CtrlSigs()).decode(inst, decode_table)

  // Exception Handling
  io.csr_decode.inst := inst
  val csr_en      = cs.csr_cmd.isOneOf(CSR.S, CSR.C, CSR.W)
  val csr_ren     = cs.csr_cmd.isOneOf(CSR.S, CSR.C) && uop.lrs1 === 0.U
  val system_insn = cs.csr_cmd === CSR.I
  val sfence      = inst === SFENCE_VMA

  val cs_legal = cs.legal
//   dontTouch(cs_legal)

  require(fLen >= 64)
  val illegal_rm = inst(14, 12).isOneOf(5.U, 6.U) || (inst(14, 12) === 7.U && io.fcsr_rm >= 5.U)

  val id_illegal_insn = !cs_legal ||
    (cs.fp_val && (io.csr_decode.fp_illegal || illegal_rm)) ||
    (uop.is_rocc && io.csr_decode.rocc_illegal) ||
    (cs.is_amo && !io.status.isa('a' - 'a')) ||
    (csr_en && (io.csr_decode.read_illegal || !csr_ren && io.csr_decode.write_illegal)) ||
    ((sfence || system_insn) && io.csr_decode.system_illegal)

//     cs.div && !csr.io.status.isa('m'-'a') || TODO check for illegal div instructions

  def checkExceptions(x: Seq[(Bool, UInt)]) =
    (x.map(_._1).reduce(_ || _), PriorityMux(x))

  val (xcpt_valid, xcpt_cause) = checkExceptions(List(
    (io.interrupt && !io.enq.uop.is_sfb, io.interrupt_cause), // Disallow interrupts while we are handling a SFB
    (uop.bp_debug_if, (CSR.debugTriggerCause).U),
    (uop.bp_xcpt_if, (Causes.breakpoint).U),
    (uop.xcpt_pf_if, (Causes.fetch_page_fault).U),
    (uop.xcpt_ae_if, (Causes.fetch_access).U),
    (id_illegal_insn, (Causes.illegal_instruction).U)
  ))

  uop.exception := xcpt_valid
  uop.exc_cause := xcpt_cause

  //-------------------------------------------------------------

  uop.is_mov := inst === ADD && LRS1 === 0.U

  uop.iq_type(IQ_UNQ) := Seq(FC_MUL, FC_DIV, FC_CSR, FC_I2F).map(c => cs.fu_code(c)).reduce(_ || _)
  uop.iq_type(IQ_ALU) := Seq(FC_ALU).map(c => cs.fu_code(c)).reduce(_ || _)
  uop.iq_type(IQ_MEM) := Seq(FC_AGEN, FC_DGEN).map(c => cs.fu_code(c)).reduce(_ || _)
  uop.iq_type(IQ_FP)  := Seq(FC_FPU, FC_FDV, FC_F2I).map(c => cs.fu_code(c)).reduce(_ || _)

  uop.fu_code := cs.fu_code.asBools

  uop.ldst := LDST
  uop.lrs1 := LRS1
  uop.lrs2 := LRS2
  uop.lrs3 := LRS3

  uop.dst_rtype  := cs.dst_type
  uop.lrs1_rtype := Mux(cs.rs1_type === RT_FIX && LRS1 === 0.U, RT_ZERO, cs.rs1_type)
  uop.lrs2_rtype := Mux(cs.rs2_type === RT_FIX && LRS2 === 0.U, RT_ZERO, cs.rs2_type)
  uop.frs3_en    := cs.frs3_en

  uop.ldst_is_rs1 := uop.is_sfb_shadow
  // SFB optimization
  when(uop.is_sfb_shadow && cs.rs2_type === RT_X) {
    uop.lrs2_rtype  := Mux(LDST === 0.U, RT_ZERO, RT_FIX)
    uop.lrs2        := LDST
    uop.ldst_is_rs1 := false.B
  }.elsewhen(uop.is_sfb_shadow && uop.is_mov) {
    uop.lrs1        := LDST
    uop.lrs1_rtype  := Mux(LDST === 0.U, RT_ZERO, RT_FIX)
    uop.ldst_is_rs1 := true.B
  }

  uop.fp_val  := cs.fp_val
  uop.fp_ctrl := cs.fp

  uop.mem_cmd         := cs.mem_cmd
  uop.mem_size        := Mux(cs.mem_cmd.isOneOf(M_SFENCE, M_FLUSH_ALL), Cat(LRS2 =/= 0.U, LRS1 =/= 0.U), inst(13, 12))
  uop.mem_signed      := !inst(14)
  uop.uses_ldq        := cs.uses_ldq
  uop.uses_stq        := cs.uses_stq
  uop.is_amo          := cs.is_amo
  uop.is_fence        := inst === FENCE
  uop.is_fencei       := inst === FENCE_I
  uop.is_sfence       := inst === SFENCE_VMA
  uop.is_sys_pc2epc   := inst === EBREAK || inst === ECALL
  uop.is_eret         := inst === ECALL || inst === EBREAK || inst === SRET || inst === MRET || inst === DRET
  uop.is_unique       := cs.inst_unique || uop.is_rocc
  uop.is_rocc         := inst(6, 0).isOneOf("b0001011".U, "b0101011".U, "b1111011".U) && inst(14, 12).isOneOf(
    0.U,
    2.U,
    3.U,
    4.U,
    6.U,
    7.U
  )
  uop.flush_on_commit := cs.flush_on_commit || (csr_en && !csr_ren && io.csr_decode.write_flush)

  //-------------------------------------------------------------
  // immediates

  // repackage the immediate, and then pass the fewest number of bits around
  val di24_20    = Mux(cs.imm_sel === IS_B || cs.imm_sel === IS_S, inst(11, 7), inst(24, 20))
  val imm_packed = Cat(inst(31, 25), di24_20, inst(19, 12))
  val imm        = ImmGen(imm_packed, cs.imm_sel)
  val imm_hi     = imm >> (immPregSz - 1)
  val imm_lo     = imm(immPregSz - 1, 0)
  val short_imm  = imm_hi === 0.U || ~imm_hi === 0.U || cs.imm_sel === IS_F3

  uop.imm_rename := cs.imm_sel =/= IS_N && cs.imm_sel =/= IS_F3
  uop.imm_packed := imm_packed
  uop.imm_sel    := cs.imm_sel
  when(short_imm) {
    uop.imm_rename := false.B
    uop.imm_sel    := IS_SH
    uop.pimm       := Mux(cs.imm_sel === IS_F3, inst(14, 12), imm_lo)
  }

  uop.fp_rm  := Mux(inst(14, 12) === 7.U, io.fcsr_rm, inst(14, 12))
  uop.fp_typ := inst(21, 20)

  //-------------------------------------------------------------

  uop.csr_cmd := cs.csr_cmd
  when((cs.csr_cmd === CSR.S || cs.csr_cmd === CSR.C) && LRS1 === 0.U) {
    uop.csr_cmd := CSR.R
  }

  uop.fcn_dw := cs.fcn_dw
  uop.fcn_op := cs.fcn_op

  uop.op1_sel := OP1_RS1
  when(inst === LUI || inst === CSRRWI || inst === CSRRSI || inst === CSRRCI ||
    inst === WFI || inst === SRET || inst === MRET || inst === DRET) {
    uop.op1_sel := OP1_ZERO
  }.elsewhen(inst === JAL || inst === JALR || inst === AUIPC) {
    uop.op1_sel := OP1_PC
  }.elsewhen(Seq(SH1ADD, SH2ADD, SH3ADD, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW, ADD_UW, SLLI_UW).map(_ === inst).orR) {
    uop.op1_sel := OP1_RS1SHL
  }

  uop.op2_sel := OP2_RS2
  when(cs.is_amo || inst === CSRRW || inst === CSRRS || inst === CSRRC) {
    uop.op2_sel := OP2_ZERO
  }.elsewhen(inst === CSRRWI || inst === CSRRSI || inst === CSRRCI ||
    inst === WFI || inst === SRET || inst === DRET || inst === MRET) {
    uop.op2_sel := OP2_IMMC
  }.elsewhen(inst === JAL || inst === JALR) {
    uop.op2_sel := OP2_NEXT
  }.elsewhen(Seq(BCLR, BCLRI, BINV, BINVI, BSET, BSETI).map(_ === inst).orR) {
    uop.op2_sel := Mux(uop.lrs2_rtype === RT_FIX, OP2_RS2OH, OP2_IMMOH)
  }.elsewhen(cs.imm_sel === IS_U || cs.imm_sel === IS_I || cs.imm_sel === IS_S) {
    uop.op2_sel := OP2_IMM
  }

  uop.br_type := Seq(
    (BEQ, B_EQ),
    (BNE, B_NE),
    (BGE, B_GE),
    (BGEU, B_GEU),
    (BLT, B_LT),
    (BLTU, B_LTU),
    (JAL, B_J),
    (JALR, B_JR)
  ).map { case (c, b) => Mux(inst === c, b, 0.U) }.reduce(_ | _)

  io.deq.uop  := uop
}
