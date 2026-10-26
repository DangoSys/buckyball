package framework.core.clink

import framework.core.clink.configs.{Params => CLinkParam}
import framework.ip.bus.chi
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.memdomain.{MvoverPort, PhysicalBankHash}
import framework.thread.tlink.BankAccess

import chisel3._
import chisel3.util._
import framework.ip.bus.axi4
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

class SharedPort(axi: axi4.Params) extends Bundle {
  val tx = new axi4.Port(axi)
  val rx = Flipped(new axi4.Port(axi))
}

/** An explicitly attached shared-bank fabric; it does not imply hardware threads. */
class CoreBanks(geometry: Option[(FrontendParam, MemDomainParam, Int)], diff: Boolean) extends Bundle {
  val access = geometry.map { case (frontend, memory, ports) => new BankAccess(frontend, memory) }
  val move   = new MvoverPort

  val barrier = Option.when(geometry.nonEmpty)(new Bundle {
    val arrive  = Output(Bool())
    val release = Input(Bool())
  })

  val hashes = geometry.filter(_ => diff).map { case (frontend, memory, _) =>
    Input(Vec(memory.sharedBankNum, new PhysicalBankHash(frontend)))
  }

}

/** Core memory protocols and hardware interrupt notifications behind CLink. */
class CLinkIO(
  val p:         CLinkParam,
  bankInterface: Option[CoreBanks] = None)
    extends Bundle {

  val banks = bankInterface.map(_.cloneType)

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

  val c2c = p.c2c.map(new SharedPort(_))
}

class CoreInterrupts extends Bundle {
  val timer              = Bool()
  val software           = Bool()
  val external           = Bool()
  val supervisorExternal = Bool()
}
