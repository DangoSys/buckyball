package framework.frontend.decoder

import chisel3._
import chisel3.util._

object GISA {
  // enable=000, opcode group for no-bank-access instructions
  val MVIN_KERNEL_BITPAT = BitPat("b0101100") // 44: writes program bank
  val RUN_KERNEL_BITPAT  = BitPat("b1001111") // 79: read data/program, write data
  val BARRIER_BITPAT     = BitPat("b0000001") // 1 (0x01) — enable=000, opcode=1
}

// Domain ID constants
object DomainId {
  val FRONTEND = 0.U(4.W) // Frontend (barrier), does not enter ROB queue
  val RVV      = 2.U(4.W) // Private kernel execution and image control
  val MEM      = 1.U(4.W) // Memory domain
  val BALL     = 3.U(4.W) // Ball domain
}
