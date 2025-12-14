package framework.gpdomain

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

  // RVV Vector Compute Instructions (opcode 0x57, func3=0-6)
  // func7 field [31:25] for vector arithmetic instructions
  val VADD_VV    = BitPat("b0000000") // 0x00: vadd.vv
  val VADD_VX    = BitPat("b0000000") // 0x00: vadd.vx
  val VADD_VI    = BitPat("b0000000") // 0x00: vadd.vi
  val VSUB_VV    = BitPat("b0000010") // 0x02: vsub.vv
  val VSUB_VX    = BitPat("b0000010") // 0x02: vsub.vx
  val VRSUB_VX   = BitPat("b0000011") // 0x03: vrsub.vx
  val VRSUB_VI   = BitPat("b0000011") // 0x03: vrsub.vi

  val VMINU_VV   = BitPat("b0000100") // 0x04: vminu.vv
  val VMINU_VX   = BitPat("b0000100") // 0x04: vminu.vx
  val VMIN_VV    = BitPat("b0000101") // 0x05: vmin.vv
  val VMIN_VX    = BitPat("b0000101") // 0x05: vmin.vx
  val VMAXU_VV   = BitPat("b0000110") // 0x06: vmaxu.vv
  val VMAXU_VX   = BitPat("b0000110") // 0x06: vmaxu.vx
  val VMAX_VV    = BitPat("b0000111") // 0x07: vmax.vv
  val VMAX_VX    = BitPat("b0000111") // 0x07: vmax.vx

  val VAND_VV    = BitPat("b0001001") // 0x09: vand.vv
  val VAND_VX    = BitPat("b0001001") // 0x09: vand.vx
  val VAND_VI    = BitPat("b0001001") // 0x09: vand.vi
  val VOR_VV     = BitPat("b0001010") // 0x0A: vor.vv
  val VOR_VX     = BitPat("b0001010") // 0x0A: vor.vx
  val VOR_VI     = BitPat("b0001010") // 0x0A: vor.vi
  val VXOR_VV    = BitPat("b0001011") // 0x0B: vxor.vv
  val VXOR_VX    = BitPat("b0001011") // 0x0B: vxor.vx
  val VXOR_VI    = BitPat("b0001011") // 0x0B: vxor.vi

  val VSLL_VV    = BitPat("b0010101") // 0x15: vsll.vv
  val VSLL_VX    = BitPat("b0010101") // 0x15: vsll.vx
  val VSLL_VI    = BitPat("b0010101") // 0x15: vsll.vi
  val VSRL_VV    = BitPat("b0010100") // 0x14: vsrl.vv
  val VSRL_VX    = BitPat("b0010100") // 0x14: vsrl.vx
  val VSRL_VI    = BitPat("b0010100") // 0x14: vsrl.vi
  val VSRA_VV    = BitPat("b0010101") // 0x15: vsra.vv
  val VSRA_VX    = BitPat("b0010101") // 0x15: vsra.vx
  val VSRA_VI    = BitPat("b0010101") // 0x15: vsra.vi

  val VMUL_VV    = BitPat("b1001010") // 0x4A: vmul.vv
  val VMUL_VX    = BitPat("b1001010") // 0x4A: vmul.vx
  val VMULH_VV   = BitPat("b1001011") // 0x4B: vmulh.vv
  val VMULH_VX   = BitPat("b1001011") // 0x4B: vmulh.vx
  val VMULHU_VV  = BitPat("b1001000") // 0x48: vmulhu.vv
  val VMULHU_VX  = BitPat("b1001000") // 0x48: vmulhu.vx
  val VMULHSU_VV = BitPat("b1001001") // 0x49: vmulhsu.vv
  val VMULHSU_VX = BitPat("b1001001") // 0x49: vmulhsu.vx

  val VMACC_VV   = BitPat("b1011010") // 0x5A: vmacc.vv
  val VMACC_VX   = BitPat("b1011010") // 0x5A: vmacc.vx
  val VMADD_VV   = BitPat("b1010010") // 0x52: vmadd.vv
  val VMADD_VX   = BitPat("b1010010") // 0x52: vmadd.vx

  // RVV Vector Load/Store Instructions
  // mop field [27:26] for memory operations
  val VLE_UNIT   = BitPat("b00") // unit-stride load
  val VLE_STRIDED = BitPat("b10") // strided load
  val VLE_INDEXED = BitPat("b11") // indexed load
  val VSE_UNIT   = BitPat("b00") // unit-stride store
  val VSE_STRIDED = BitPat("b10") // strided store
  val VSE_INDEXED = BitPat("b11") // indexed store

  // Vector configuration instructions (vsetvl, vsetvli, vsetivli)
  val VSETVLI    = BitPat("b0000000") // vsetvli
  val VSETIVLI   = BitPat("b1100000") // vsetivli
  val VSETVL     = BitPat("b1000000") // vsetvl
}
