package framework.core.controller.admission

import chisel3._

/** Decoded management instruction issued by the execution core. */
class Management extends Bundle {
  val operation = UInt(7.W)
  val context   = UInt(32.W)
  val field     = UInt(32.W)
  val data      = UInt(64.W)
}
