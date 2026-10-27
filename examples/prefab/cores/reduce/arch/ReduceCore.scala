package examples.core.reduce

import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.clink.{CLinkIO, HasCLink}
import framework.device.cbus.{C2CTransfer, Control, SharedAxiEndpoint}
import framework.core.controller.admission.DirectAdmission
import framework.core.tbus.SharedNetwork
import framework.core.tbus.configs.Layout
import framework.core.shm.SHM
import framework.acc.debug.Debug
import framework.ip.bus.chi.rnf.AxiCache

/** Native execution core with a direct BBus accelerator and shared-bank C2C access. */
@instantiable
class ReduceCore(corep: CoreParam, simp: SimParam) extends Module with HasCLink {
  val diffTest = simp.diffTest
  val accp     = corep.acc.get
  require(corep.threads.isEmpty && !corep.clink.backing && corep.clink.requesters == 3)
  require(corep.clink.shared && accp.memDomain.sharedEnable && !accp.gpDomain.enable)
  require(corep.transferLocalAccess && corep.transferLeaseAccess && corep.tbus.get.router.physicalPorts == 4)
  @public
  val clink    = IO(new CLinkIO(corep.clink))

  val loom:      Instance[Loom]            = Instantiate(new Loom(corep))
  val admission: Instance[DirectAdmission] = Instantiate(new DirectAdmission(corep.controller.commands))

  val acc: Instance[Acc] = Instantiate(new Acc(
    accp,
    corep.privileged.permission,
    corep.memory.dcache.tracking,
    corep.memory.memoryAccess.walker,
    simp
  ))

  loom.io.interrupts        := clink.interrupts(0)
  admission.io.execution <> loom.io.commands
  loom.io.drained           := admission.io.drained
  admission.io.accelerator <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  clink.memory.uncached <> loom.io.uncached
  clink.memory.coherent.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }

  val dma = Instantiate(new AxiCache(accp.transport.axi, corep.dmaCache.get))
  dma.io.axi <> acc.io.ddr
  dma.io.axi.ar.bits.cache  := 15.U
  dma.io.axi.aw.bits.cache  := 15.U
  clink.memory.coherent(2) <> dma.io.chi
  clink.memory.ddr.aw.valid := false.B
  clink.memory.ddr.aw.bits  := 0.U.asTypeOf(clink.memory.ddr.aw.bits)
  clink.memory.ddr.w.valid  := false.B
  clink.memory.ddr.w.bits   := 0.U.asTypeOf(clink.memory.ddr.w.bits)
  clink.memory.ddr.ar.valid := false.B
  clink.memory.ddr.ar.bits  := 0.U.asTypeOf(clink.memory.ddr.ar.bits)
  clink.memory.ddr.b.ready  := false.B
  clink.memory.ddr.r.ready  := false.B

  val banks =
    Instantiate(new SharedNetwork(
      accp.frontend,
      accp.memDomain,
      corep.tbus.get,
      Seq(corep.hartId),
      Seq(corep.hartId),
      simp
    ))

  val shm = Instantiate(new SHM(accp.frontend, accp.memDomain, corep.tbus.get))
  shm.io <> banks.io.shm
  banks.io.threads(0).bank <> acc.io.shm.bank
  banks.io.threads(0).move <> acc.io.shm.move
  banks.io.threads(0).barrierArrive := acc.io.shm.barrierArrive
  acc.io.shm.barrierRelease         := banks.io.threads(0).barrierRelease
  acc.io.shared_bank_hashes.zip(banks.io.bankHashes).foreach { case (sink, source) => sink := source }

  val control = Instantiate(new Control(accp.memDomain, corep.c2c, corep.transferLeaseAccess, corep.transferLocalAccess))
  val transfer = Instantiate(new C2CTransfer(corep.c2c))
  val endpoint = Instantiate(new SharedAxiEndpoint(corep.c2c))
  control.io.coreId        := corep.hartId.U
  control.io.request <> admission.io.remote.request
  admission.io.remote.reply <> control.io.reply
  admission.io.remote.busy := !control.io.idle || endpoint.io.busy
  transfer.io.command <> control.io.command
  control.io.completion <> transfer.io.completion
  banks.io.external.get.memory(0) <> transfer.io.source
  banks.io.external.get.memory(1) <> endpoint.io.read
  banks.io.external.get.memory(2) <> endpoint.io.write
  banks.io.external.get.memory(3) <> control.io.local.get
  banks.io.external.get.lease.get <> control.io.lease.get
  clink.c2c.get.tx <> transfer.io.mem
  endpoint.io.mem <> clink.c2c.get.rx

  val debug = Instantiate(new Debug(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    simp,
    corep.hartId,
    corep.hartId
  ))

  debug.io.ant.valid := false.B
  debug.io.ant.bits  := 0.U.asTypeOf(debug.io.ant.bits)
  debug.io.acc <> acc.io.debug
}

object Reduce extends CoreFactory {
  val component = "reduce"
  def parameters(design: Design, source: CorePlacement): CoreParam =
    CoreParam.fromProto(design, source, externalHome = true)

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new ReduceCore(corep, simp)).clink

}
