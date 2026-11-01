package framework.core.shm

import chisel3._
import chisel3.util._
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}
import framework.core.tbus.router.{Lease, PhysicalAccess}
import framework.acc.memdomain.{BankPort, LayoutConfig, MvoverISA, MvoverPort}
import framework.acc.memdomain.bank_router.BankMapping

/** Core-local shared-bank access, lifetime and synchronization. */
class BankAccess(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val sharedPorts = memParam.sharedPorts
  require(sharedPorts >= 0 && (memParam.sharedEnable || sharedPorts == 0))
  val requests    = Vec(sharedPorts, new BankPort(frontendParam, memParam))

  val local = Flipped(new LocalBank(
    MvoverISA.AddressBits,
    MvoverISA.BankBits,
    memParam.bankWidth,
    math.max(1, log2Ceil(frontendParam.rob_entries))
  ))

  val config  = Decoupled(new LayoutConfig(frontendParam, memParam))
  val mapping = Decoupled(new BankMapping(frontendParam))
}

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
