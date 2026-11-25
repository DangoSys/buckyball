package framework.generic.contract.frontend

import chisel3._
import chisel3.util._
import framework.generic.contract.memory.access.preflight.PmpEntry
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.Command
import framework.generic.contract.frontend.Response
import framework.generic.contract.frontend.RoCCCommand
import framework.generic.contract.frontend.RoCCIO

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
