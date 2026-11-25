package framework.generic.contract.memory.access.preflight

import chisel3._
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.memory.access.preflight.PmpConfig
import framework.generic.contract.memory.access.preflight.PmpEntry

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
