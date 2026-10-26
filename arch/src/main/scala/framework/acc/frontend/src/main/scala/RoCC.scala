package framework.acc.frontend

import chisel3._
import chisel3.util._
import framework.core.memory.access.preflight.PmpEntry
import framework.core.memory.access.preflight.configs.PmpParams

class Response(dataBits: Int) extends Bundle {

  val rd   = UInt(5.W)
  val data = UInt(dataBits.W)
}

/** RoCC instruction operands and the frozen submitting address-space context. */
class RoCCCommand(dataBits: Int, permission: PmpParams) extends Bundle {
  val instruction        = new Command(dataBits)
  val satp               = UInt(64.W)
  val effectivePrivilege = UInt(2.W)
  val sum                = Bool()
  val mxr                = Bool()
  val pmp                = Vec(permission.entries, new PmpEntry(permission))
}

/** Accelerator-facing RoCC direction; the host uses Flipped. */
class RoCCIO(dataBits: Int, permission: PmpParams) extends Bundle {
  val cmd       = Flipped(Decoupled(new RoCCCommand(dataBits, permission)))
  val resp      = Decoupled(new Response(dataBits))
  val busy      = Output(Bool())
  val interrupt = Output(Bool())
}
