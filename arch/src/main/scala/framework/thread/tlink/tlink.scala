package framework.thread.tlink

import sims.configs.SimParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}

import chisel3._
import chisel3.util._
import framework.core.tbus.router.{LeasePort, PhysicalPort}
import framework.acc.memdomain.{BankPort, LayoutConfig, MvoverISA, MvoverPort}
import framework.acc.memdomain.bank_router.BankMapping
import framework.ip.memory.mesh_shm.MeshLocalBankPort
import framework.thread.configs.ThreadParam
import framework.thread.ant.Task
import framework.core.memory.MemoryPort
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.acc.memdomain.PhysicalBankHash
import framework.ip.memory.spm
import framework.ip.bus.axi4

/** Thread-facing shared-bank access, lifetime and synchronization. */
class BankAccess(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val sharedPorts = memParam.sharedPorts
  require(sharedPorts >= 0 && (memParam.sharedEnable || sharedPorts == 0))
  val requests    = Vec(sharedPorts, new BankPort(frontendParam, memParam))

  val local = Flipped(new MeshLocalBankPort(
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

/** Complete thread boundary; each service retains its independent handshake. */
class TLinkIO(
  thread:     ThreadParam,
  permission: PmpParams,
  tracking:   TrackingParam,
  walker:     WalkerParam,
  simp:       SimParam)
    extends Bundle {
  val sharedPorts = thread.acc.get.memDomain.sharedPorts
  val diffTest    = simp.diffTest
  val task        = new Task(thread.ant.get, permission)
  val tss         = new spm.Port(thread.ant.get.shared)
  val memory      = new MemoryPort(tracking, walker)
  val ddr         = new axi4.Port(thread.acc.get.transport.axi)
  val shm         = new SharedLink(thread.acc.get.frontend, thread.acc.get.memDomain)

  val sharedHashes = Option.when(diffTest && thread.acc.get.memDomain.sharedEnable)(
    Input(Vec(thread.acc.get.memDomain.sharedBankNum, new PhysicalBankHash(thread.acc.get.frontend)))
  )

}

/** External shared-memory access and optional bank-movement commands. */
class ExternalLink(memParam: MemDomainParam, tbusParam: TBusParam) extends Bundle {
  val p     = tbusParam
  val lease = Option.when(p.router.physicalPorts > 0)(Flipped(new LeasePort))

  val memory = Vec(
    p.router.physicalPorts,
    Flipped(new PhysicalPort(memParam.bankWidth))
  )

  val move = Option.when(p.externalMove)(Flipped(new MvoverPort))
}
