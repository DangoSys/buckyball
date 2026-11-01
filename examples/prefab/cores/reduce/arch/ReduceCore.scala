package examples.core.reduce

import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.clink.{CLinkIO, HasCLink}
import framework.core.controller.admission.DirectAdmission
import framework.core.tbus.{Barrier, SharedNetwork}
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
  require(corep.clink.service && accp.memDomain.sharedEnable && !accp.gpDomain.enable)
  require(corep.tbus.get.router.physicalPorts == 0)
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
      simp,
      0
    ))

  val shm     = Instantiate(new SHM(accp.frontend, accp.memDomain, corep.tbus.get))
  val barrier = Instantiate(new Barrier(corep.tbus.get))
  shm.io <> banks.io.shm
  banks.io.banks(0) <> acc.io.shm.bank
  banks.io.moves(0) <> acc.io.shm.move
  barrier.io.arrive(0)      := acc.io.shm.barrierArrive
  acc.io.shm.barrierRelease := barrier.io.release(0)
  acc.io.shared_bank_hashes.zip(banks.io.bankHashes).foreach { case (sink, source) => sink := source }

  clink.service.get <> admission.io.remote

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
