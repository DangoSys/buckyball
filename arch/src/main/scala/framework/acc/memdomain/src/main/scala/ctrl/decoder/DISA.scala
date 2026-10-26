package framework.acc.memdomain

import chisel3._
import chisel3.util._

object DISA {
  val MVOVER_BITPAT      = BitPat("b0001101")
  val MVIN_MMIO_BITPAT   = BitPat("b0100011")
  val MSET_FUNCT         = 0x20
  // enable=010 (1 write), opcode group
  val MSET_BITPAT        = BitPat("b0100000") // 32 (0x20) — enable=010, opcode=0
  val MVIN_BITPAT        = BitPat("b0100001") // 33 (0x21) — enable=010, opcode=1
  val MVIN_2D_BITPAT     = BitPat("b0100010") // 34 (0x22) — 2D DMA into bank rows
  val MVIN_KERNEL_BITPAT = BitPat("b0101100")
  // enable=001 (1 read), opcode group
  val MVOUT_BITPAT       = BitPat("b0010000") // 16 (0x10) — enable=001, opcode=0
}
