package framework.generic.contract.frontend.decoder

import chisel3._
import chisel3.util._
import framework.generic.contract.frontend.decoder.DomainId

// Domain ID constants
object DomainId {
  val FRONTEND = 0.U(4.W) // Frontend (barrier), does not enter ROB queue
  val GP       = 2.U(4.W) // Private operator execution and image control
  val MEM      = 1.U(4.W) // Memory domain
  val BALL     = 3.U(4.W) // Ball domain
}
