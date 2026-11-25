package examples.designs.goban.harts.goban

import framework.thread.configs.ThreadParam

import framework.thread.tbus.TBus

import framework.thread.tbus.generated.ThreadComponents
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.rocket.configs.{Params => RocketParam}
import framework.hart.configs.HartParam
import framework.tlp.controller.Controller
import framework.tlp.shm.{MeshSharedMemParams, MeshStorage, MeshThreadAttachment}
import framework.hart.memory.{Requester}
import framework.tlp.memory.{Dataflow, DmaMemory}
import framework.tlp.tss.Store
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.generic.contract.sims.configs.SimParam

@instantiable
class GobanHart(
  hartp:    HartParam,
  threads:  Seq[ThreadParam],
  rocketp:  RocketParam,
  simp:     SimParam,
  traceIds: Seq[Int])
    extends Module
    with HasHLink {
  val diffTest = simp.diffTest
  require(hartp.hlink.harts == 1 && !hartp.hlink.backing && !hartp.hlink.service)
  require(hartp.hlink.requesters == 1 && hartp.threads.nonEmpty)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom:       Instance[Loom]       = Instantiate(new Loom(hartp, rocketp))
  val controller: Instance[Controller] = Instantiate(new Controller(hartp.controller))

  val transport = Instantiate(new DmaMemory(
    hartp.interconnect.bus,
    hartp.privileged.permission,
    hartp.memory,
    hartp.hlink.line.addressBits,
    hartp.hlink.backing,
    hartp.threads.size
  ))

  val tbus: Instance[TBus] =
    Instantiate(new TBus(
      threads,
      hartp.privileged.permission,
      hartp.tss.get.memory,
      TBus.generators(
        threads,
        hartp.privileged.permission,
        hartp.hartId,
        simp,
        traceIds,
        ThreadComponents.factories
      )
    ))

  val storage = hartp.shm.get

  val geometry = MeshSharedMemParams(
    rows = 2,
    cols = 2,
    entriesPerBank = storage.entriesPerBank,
    dataBits = storage.dataBits,
    tagBits = 2,
    banksPerNode = storage.banks / 4,
    threads = Seq(0, 1, 2, 3).map(node => MeshThreadAttachment(node, 2))
  )

  val shm      = Instantiate(new MeshStorage(geometry, transfers = false))
  val dataflow = Instantiate(new Dataflow(geometry, hartp.interconnect))
  shm.io.channels <> dataflow.io.storage
  shm.io.transferCommand.valid      := false.B
  shm.io.transferCommand.bits       := 0.U.asTypeOf(shm.io.transferCommand.bits)
  shm.io.transferCompletion.ready   := true.B
  shm.io.localBanks.foreach { port =>
    port.request.ready  := false.B
    port.response.valid := false.B
    port.response.bits  := 0.U.asTypeOf(port.response.bits)
  }
  shm.io.bankWrites.foreach(_.ready := true.B)

  val tss: Instance[Store] = Instantiate(new Store(hartp.tss.get))

  loom.io.interrupts                  := hlink.interrupts(0)
  controller.io.execution <> loom.io.commands
  loom.io.memoryAllowed               := controller.io.memoryAllowed
  tbus.io.control.get <> controller.io.threads.get
  loom.io.memory <> transport.io.services
  hlink.memory.page.get <> transport.io.page
  controller.io.move.command.ready    := false.B
  controller.io.move.completion.valid := false.B
  controller.io.move.completion.bits  := false.B
  dataflow.io.command <> controller.io.dataflow.get.request
  controller.io.dataflow.get.reply <> dataflow.io.reply
  controller.io.dataflow.get.busy     := dataflow.io.busy
  controller.io.stalled.get           := dataflow.io.stalled
  for (thread <- hartp.threads.indices) {
    dataflow.io.threads(thread) <> tbus.io.dma(thread)
    transport.io.clients(thread) <> dataflow.io.external(thread)
    transport.io.launch(thread).valid := controller.io.threads.get(thread).start.fire
    transport.io.launch(thread).bits  := controller.io.threads.get(thread).start.bits.context
    dataflow.io.task(thread).valid    := controller.io.threads.get(thread).start.fire
    dataflow.io.task(thread).bits     := controller.io.threads.get(thread).start.bits.descriptor.task
  }
  tss.io.clients <> tbus.io.tss.get
  tss.io.host <> controller.io.tss.get.host
  tss.io.inUse := controller.io.tss.get.inUse
  tss.io.cancel                  := controller.io.tss.get.cancel
  controller.io.tss.get.busy     := tss.io.busy
  controller.io.tss.get.hostBusy := tss.io.hostBusy

  hlink.memory.uncached <> loom.io.uncached

  val requester = Instantiate(new Requester(
    hartp.hlink.line,
    hartp.hlink.requesterIds.head,
    Seq(hartp.memory.icache.nodeId, hartp.memory.dcache.cache.nodeId),
    2
  ))

  requester.io.clients.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  hlink.memory.coherent(0) <> requester.io.coherent

  val memory = Instantiate(new framework.generic.bus.axi4.Interconnect(
    framework.generic.bus.axi4.InterconnectParam(hartp.interconnect.bus, hartp.threads.size, None)
  ))

  memory.io.in <> transport.io.axi
  hlink.memory.ddr <> memory.io.out
}

object Goban extends HartFactory {
  val component = "goban"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    val hart = HartParam.fromProto(design, source, externalHome = !RocketParam.fromProto(source).hlink.backing)
    require(hart.threads.size == 4 && hart.shm.get.banks == 16)
    hart
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(
      new GobanHart(hartp, ThreadParam.ofHart(design, source), RocketParam.fromProto(source), simp, traceIds)
    ).hlink

}
