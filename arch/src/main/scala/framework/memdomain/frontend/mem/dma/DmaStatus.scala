package framework.memdomain.frontend.mem.dma

import chisel3._

class DmaStatus extends Bundle {
  val error   = UInt(4.W)
  val address = UInt(64.W)
}

object DmaError {
  val None        = 0
  val PageFault   = 1
  val AccessFault = 2
  val Denied      = 3
  val Corrupt     = 4
  val Bank        = 5
  val Protocol    = 6
  val Shape       = 7
  val Capacity    = 8
  val Context     = 9
  val Cancelled   = 10
}
