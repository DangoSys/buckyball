package framework.memdomain.backend.shared

import chisel3._
import chisel3.util._

class SharedLeaseRequest extends Bundle {
  val endpoint = UInt(32.W)
  val vbank    = UInt(16.W)
  val group    = UInt(16.W)
  val release  = Bool()
}

/** One outstanding mapping lease operation; export returns its physical byte base. */
class SharedLeasePort extends Bundle {
  val request  = Decoupled(new SharedLeaseRequest)
  val response = Flipped(Decoupled(UInt(64.W)))
}
