package framework.tlp.shm

import framework.generic.acc.shm.BankAccess
import framework.generic.acc.memdomain.LayoutConfig
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort
import framework.generic.acc.memdomain.bank_router.BankMapping
import chisel3._
import chisel3.util._
import framework.tlp.tbus.configs.{Params => TBusParam}
import framework.tlp.tbus.router.{Lease, PhysicalAccess}
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.memdomain.BankPort

class SharedLink(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val sharedPorts    = memParam.sharedPorts
  val bank           = new BankAccess(frontendParam, memParam)
  val move           = new MvoverPort
  val barrierArrive  = Output(Bool())
  val barrierRelease = Input(Bool())
}

/** External shared-memory access and optional bank-movement commands. */
class ExternalLink(memParam: MemDomainParam, tbusParam: TBusParam) extends Bundle {
  val p     = tbusParam
  val lease = Option.when(p.router.physicalPorts > 0)(Flipped(new Lease))

  val memory = Vec(
    p.router.physicalPorts,
    Flipped(new PhysicalAccess(memParam.bankWidth))
  )

  val move = Option.when(p.externalMove)(Flipped(new MvoverPort))
}
