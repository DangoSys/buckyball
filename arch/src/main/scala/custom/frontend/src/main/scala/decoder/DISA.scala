package framework.custom.frontend.decoder

import chisel3._
import chisel3.util._

object DISA {
  val BARRIER_BITPAT = BitPat("b0000001") // 1 (0x01) — enable=000, opcode=1
}
