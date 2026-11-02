package examples.designs.goban

import framework.core.tbus.generated.ThreadComponents
import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.rocket.configs.{Params => RocketParam}
import framework.core.configs.CoreParam
import framework.core.controller.Controller
import framework.core.tbus.TBus
import framework.core.shm.{Exchange, MeshSharedMemParams, MeshThreadAttachment}
import framework.core.tss.Store
import framework.core.clink.{CLinkIO, HasCLink}
import framework.ip.bus.chi.rnf.AxiCache

@instantiable
class GobanCore(
  corep:    CoreParam,
  rocketp:  RocketParam,
  simp:     SimParam,
  traceIds: Seq[Int])
    extends Module
    with HasCLink {
  val diffTest = simp.diffTest
  require(corep.clink.harts == 1 && !corep.clink.backing && !corep.clink.service)
  require(corep.clink.requesters == 3 && corep.threads.nonEmpty)

  @public
  val clink = IO(new CLinkIO(corep.clink))

  val loom:       Instance[Loom]       = Instantiate(new Loom(corep, rocketp))
  val controller: Instance[Controller] = Instantiate(new Controller(corep.controller))

  val transport = Instantiate(new framework.core.memory.DmaMemory(corep))
  val tbus: Instance[TBus] =
    Instantiate(new TBus(corep, simp, traceIds, TBus.generators(corep, simp, traceIds, ThreadComponents.factories)))

  val storage = corep.shm.get

  val shm = Instantiate(new Exchange(
    MeshSharedMemParams(
      rows = 2,
      cols = 2,
      entriesPerBank = storage.entriesPerBank,
      dataBits = storage.dataBits,
      tagBits = 2,
      banksPerNode = storage.banks / 4,
      threads = Seq(0, 1, 2, 3).map(node => MeshThreadAttachment(node, 2))
    ),
    corep.interconnect
  ))

  val tss: Instance[Store]    = Instantiate(new Store(corep.tss.get))
  val dma: Instance[AxiCache] = Instantiate(new AxiCache(corep.interconnect.bus, corep.dmaCache.get))

  loom.io.interrupts                  := clink.interrupts(0)
  controller.io.execution <> loom.io.commands
  loom.io.memoryAllowed               := controller.io.memoryAllowed
  tbus.io.control.get <> controller.io.threads.get
  loom.io.memory <> transport.io.services
  controller.io.move.command.ready    := false.B
  controller.io.move.completion.valid := false.B
  controller.io.move.completion.bits  := false.B
  shm.io.command <> controller.io.dataflow.get.request
  controller.io.dataflow.get.reply <> shm.io.reply
  controller.io.dataflow.get.busy     := shm.io.busy
  for (thread <- corep.threads.indices) {
    shm.io.threads(thread) <> tbus.io.dma(thread)
    transport.io.clients(thread) <> shm.io.external(thread)
    transport.io.launch(thread).valid := controller.io.threads.get(thread).start.fire
    transport.io.launch(thread).bits  := controller.io.threads.get(thread).start.bits.context
    shm.io.task(thread).valid         := controller.io.threads.get(thread).start.fire
    shm.io.task(thread).bits          := controller.io.threads.get(thread).start.bits.descriptor.task
  }
  tss.io.clients <> tbus.io.tss.get
  tss.io.host <> controller.io.tss.get.host
  tss.io.inUse := controller.io.tss.get.inUse
  tss.io.cancel                  := controller.io.tss.get.cancel
  controller.io.tss.get.busy     := tss.io.busy
  controller.io.tss.get.hostBusy := tss.io.hostBusy

  clink.memory.uncached <> loom.io.uncached
  clink.memory.coherent.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  dma.io.axi <> transport.io.axi
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
}

object Goban extends CoreFactory {
  val component = "goban"

  def parameters(design: Design, source: CorePlacement): CoreParam = {
    val core = CoreParam.fromProto(design, source, externalHome = !RocketParam.fromProto(source).clink.backing)
    require(core.threads.size == 4 && core.shm.get.banks == 16)
    core
  }

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new GobanCore(corep, RocketParam.fromProto(source), simp, traceIds)).clink

}
