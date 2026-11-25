package framework.generic.contract.hlink

import framework.generic.bus.chi
import chisel3._
import chisel3.util._
import framework.generic.bus.axi4
import framework.generic.bus.chi.snf.{LineRequest, LineResponse}
import framework.generic.memory.coherence.PageQuery
import framework.generic.contract.hlink.HartInterrupts

class HartInterrupts extends Bundle {
  val timer              = Bool()
  val software           = Bool()
  val external           = Bool()
  val supervisorExternal = Bool()
}
