package framework.frontend.decoder

import chisel3._
import chisel3.util._

object GISA {
  val FENCE_BITPAT = BitPat("b0011111") // 31 (0x1F)
}

// Domain ID constants
object DomainId {
  val FRONTEND = 0.U(4.W) // Frontend (fence), does not enter ROB queue
  val MEM      = 1.U(4.W) // Memory domain
  val GP       = 2.U(4.W) // General purpose domain (T1 vector processor)
  val BALL     = 3.U(4.W) // Ball domain
}
