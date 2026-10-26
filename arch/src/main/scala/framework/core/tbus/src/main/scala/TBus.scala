package framework.core.tbus

import sims.configs.SimParam
import framework.core.tbus.configs.Layout

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.thread.tlink.TLinkIO
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.thread.configs.ThreadParam
import framework.ip.memory.mesh_shm.MeshSharedMemIO
import framework.acc.memdomain.MvoverPort
import framework.core.controller.admission.TaskEndpoint
import framework.thread.ant.Task
import framework.core.memory.MemoryPort
import framework.ip.memory.spm
import framework.core.tbus.SharedNetwork
import framework.device.cbus.{Control => TransferControl, C2CTransfer, SharedAxiEndpoint}
import framework.core.clink.SharedPort
import framework.ip.bus.axi4
import framework.ip.bus.axi4.Interconnect

@instantiable
class TBus(
  corep:      CoreParam,
  simp:       SimParam,
  traceIds:   Seq[Int],
  generators: Seq[() => TLinkIO])
    extends Module {
  val diffTest = simp.diffTest
  val core     = corep
  require(traceIds.size == core.threads.size)

  @public
  val io = IO(new Bundle {

    val control = Option.when(core.threads.nonEmpty)(MixedVec(core.threads.map(thread =>
      new Task(thread.ant.get, core.privileged.permission)
    )))

    val memory = Option.when(core.threads.nonEmpty)(Vec(
      core.threads.size,
      new MemoryPort(core.memory.dcache.tracking, core.memory.memoryAccess.walker)
    ))

    val remote = Option.when(core.clink.shared)(Flipped(new TaskEndpoint))
    val move   = Flipped(new MvoverPort)
    val ddr    = new axi4.Port(core.interconnect.bus)
    val c2c    = Option.when(core.clink.shared)(new SharedPort(core.c2c.axi))

    val shm = Option.when(core.threads.nonEmpty)(Flipped(new MeshSharedMemIO(core.tbus.get.router.mesh(
      core.sharedFrontend,
      core.sharedMemory,
      core.tbus.get
    ))))

    val tss = Option.when(core.threads.nonEmpty)(Vec(core.threads.size, new spm.Port(core.tss.get.memory)))
  })

  require(generators.size == core.threads.size)
  val endpoints = generators.map(_())

  require(core.tss.get.ports == endpoints.size)
  require(core.interconnect.masters == endpoints.size)
  require(core.tbus.get.participants == core.threads.map(_.id) && core.tbus.get.slots == endpoints.size)

  val banks: Instance[SharedNetwork] = Instantiate(new SharedNetwork(
    core.sharedFrontend,
    core.sharedMemory,
    core.tbus.get,
    core.threads.map(_.id),
    traceIds,
    simp
  ))

  val ddr: Instance[Interconnect] = Instantiate(new Interconnect(core.interconnect))
  io.ddr <> ddr.io.out
  io.shm.get <> banks.io.shm

  if (core.clink.shared) {
    require(core.tbus.get.router.physicalPorts == 3 && !core.transferLocalAccess && core.transferLeaseAccess)
    val transferControl: Instance[TransferControl]   = Instantiate(new TransferControl(
      core.sharedMemory,
      core.c2c,
      core.transferLeaseAccess,
      core.transferLocalAccess
    ))
    val transfer:        Instance[C2CTransfer]       = Instantiate(new C2CTransfer(core.c2c))
    val endpoint:        Instance[SharedAxiEndpoint] = Instantiate(new SharedAxiEndpoint(core.c2c))
    transferControl.io.coreId := core.hartId.U
    transfer.io.command <> transferControl.io.command
    transferControl.io.completion <> transfer.io.completion
    banks.io.external.get.memory(0) <> transfer.io.source
    banks.io.external.get.memory(1) <> endpoint.io.read
    banks.io.external.get.memory(2) <> endpoint.io.write
    banks.io.external.get.lease.get <> transferControl.io.lease.get
    io.c2c.get.tx <> transfer.io.mem
    endpoint.io.mem <> io.c2c.get.rx
    transferControl.io.request <> io.remote.get.request
    io.remote.get.reply <> transferControl.io.reply
    io.remote.get.busy        := !transferControl.io.idle || endpoint.io.busy
  } else {
    require(core.tbus.get.router.physicalPorts == 0)

  }
  require(core.tbus.get.externalMove)
  banks.io.external.get.move.get <> io.move

  for ((link, index) <- endpoints.zipWithIndex) {
    io.control.get(index) <> link.task
    io.tss.get(index) <> link.tss
    io.memory.get(index) <> link.memory
    ddr.io.in(index) <> link.ddr
    banks.io.threads(index) <> link.shm
    link.sharedHashes.foreach(_ := banks.io.bankHashes.get)
  }
}

object TBus {

  def generators(
    corep:     CoreParam,
    simp:      SimParam,
    traceIds:  Seq[Int],
    factories: Map[String, ThreadFactory]
  ): Seq[() => TLinkIO] = {
    require(corep.threads.size == traceIds.size)
    corep.threads.zip(traceIds).map { case (threadp, traceId) =>
      () => factories(threadp.component).instantiate(corep, threadp, simp, traceId)
    }
  }

}

trait ThreadFactory {

  def instantiate(
    corep:   CoreParam,
    threadp: ThreadParam,
    simp:    SimParam,
    traceId: Int
  ): TLinkIO

}
