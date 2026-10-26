package framework.core.memory.access.preflight

import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._

/** Frozen RISC-V physical-memory protection state at the architectural boundary. */
class PmpConfig extends Bundle {
  val l   = Bool()
  val res = UInt(2.W)
  val a   = UInt(2.W)
  val x   = Bool()
  val w   = Bool()
  val r   = Bool()
}

class PmpEntry(b: PmpParams) extends Bundle {
  implicit val permission = b

  val cfg  = new PmpConfig
  val addr = UInt((permission.addressBits - 2).W)
  val mask = UInt(permission.addressBits.W)
}
