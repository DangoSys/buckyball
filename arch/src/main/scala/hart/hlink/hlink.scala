package framework.hart.hlink

import framework.hart.hlink.configs.{Params => HLinkParam}
import framework.generic.bus.chi

import chisel3._
import chisel3.util._
import framework.generic.bus.axi4
import framework.generic.contract.controller.{TaskEndpoint}
import framework.generic.bus.chi.snf.{LineRequest, LineResponse}
import framework.generic.memory.coherence.PageQuery
import framework.generic.contract.hlink.HartInterrupts

/** Hart memory protocols and hardware interrupt notifications behind HLink. */
class HLinkIO(
  val p: HLinkParam)
    extends Bundle {

  val interrupts = Input(Vec(p.harts, new HartInterrupts))

  val memory = new Bundle {

    val coherent = Vec(p.requesters, new chi.RequesterPort(p.line))

    val page = Option.when(p.pages)(new PageQuery(p.line.addressBits))

    val backing = Option.when(p.backing)(new Bundle {
      val request  = Decoupled(new LineRequest(p.line))
      val response = Flipped(Decoupled(new LineResponse(p.line)))
    })

    val uncached = new axi4.Port(p.uncached)

    val ddr = new axi4.Port(p.ddr)
  }

  val service = Option.when(p.service)(new TaskEndpoint)
  val payload = Option.when(p.service)(new ServiceData)
}
