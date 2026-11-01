package framework.core.clink

import framework.core.clink.configs.{Params => CLinkParam}
import framework.ip.bus.chi

import chisel3._
import chisel3.util._
import framework.ip.bus.axi4
import framework.core.controller.admission.TaskEndpoint
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

/** Core memory protocols and hardware interrupt notifications behind CLink. */
class CLinkIO(
  val p: CLinkParam)
    extends Bundle {

  val interrupts = Input(Vec(p.harts, new CoreInterrupts))

  val memory = new Bundle {

    val coherent = Vec(p.requesters, new chi.RequesterPort(p.line))

    val backing = Option.when(p.backing)(new Bundle {
      val request  = Decoupled(new LineRequest(p.line))
      val response = Flipped(Decoupled(new LineResponse(p.line)))
    })

    val uncached = new axi4.Port(p.uncached)

    val ddr = new axi4.Port(p.ddr)
  }

  val service = Option.when(p.service)(new TaskEndpoint)
}

class CoreInterrupts extends Bundle {
  val timer              = Bool()
  val software           = Bool()
  val external           = Bool()
  val supervisorExternal = Bool()
}
