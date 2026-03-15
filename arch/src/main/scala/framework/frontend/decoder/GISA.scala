package framework.frontend.decoder

import chisel3._
import chisel3.util._

object GISA {
  // enable=000, opcode group for no-bank-access instructions
  val FENCE_BITPAT   = BitPat("b0000000") // 0 (0x00) — enable=000, opcode=0
  val BARRIER_BITPAT = BitPat("b0000001") // 1 (0x01) — enable=000, opcode=1
}

// Domain ID constants
object DomainId {
  val FRONTEND = 0.U(4.W) // Frontend (fence), does not enter ROB queue
  val MEM      = 1.U(4.W) // Memory domain
  val GP       = 2.U(4.W) // General purpose domain (T1 vector processor)
  val BALL     = 3.U(4.W) // Ball domain
}
