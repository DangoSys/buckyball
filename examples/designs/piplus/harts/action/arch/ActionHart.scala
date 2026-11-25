package examples.designs.piplus.harts.action

import framework.thread.configs.ThreadParam

import framework.thread.tbus.TBus

import examples.designs.piplus.harts.generated.ThreadComponents
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.tlp.controller.Controller
import framework.tlp.shm.{MeshSharedMemParams, MeshStorage, MeshThreadAttachment}
import framework.hart.memory.{Requester}
import framework.tlp.memory.{Dataflow, DmaMemory}
import framework.generic.bus.axi4
import framework.tlp.tss.Store
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.generic.contract.sims.configs.SimParam

/**
 * One hart driving its hardware threads. Every thread has its own DMA channel of the bank's width;
 * the hart joins them on its way out to the DDR. The data passes no cache: the hart keeps its
 * caches and the device Home coherent with it on the control path (DmaMemory).
 */
@instantiable
class ActionHart(
  hartp:    HartParam,
  threads:  Seq[ThreadParam],
  simp:     SimParam,
  traceIds: Seq[Int])
    extends Module
    with HasHLink {
  require(hartp.hlink.harts == 1 && !hartp.hlink.backing && !hartp.hlink.service)
  require(hartp.hlink.requesters == 1 && hartp.threads.size == 4 && hartp.acc.isEmpty)

  @public
  val hlink = IO(new HLinkIO(hartp.hlink))

  val loom:       Instance[Loom]       = Instantiate(new Loom(hartp))
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

  // The SharedMem is a 2 x 2 mesh; each thread attaches at its own node.
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
  loom.io.drained                     := controller.io.drained
  loom.io.memoryAllowed               := controller.io.memoryAllowed
  tbus.io.control.get <> controller.io.threads.get
  loom.io.dma.get <> transport.io.services
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

  // Each thread's channel carries the DDR's beat of bank rows, so the threads share the DDR port
  // beat by beat, not a bank row at a time.
  val memory = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(hartp.hlink.ddr, hartp.threads.size, None)))
  for ((channel, port) <- transport.io.axi.zip(memory.io.in)) {
    val widen = Instantiate(new axi4.Widen(hartp.interconnect.bus, hartp.hlink.ddr, 256))
    widen.io.in <> channel
    port <> widen.io.out
  }
  hlink.memory.ddr <> memory.io.out
}

/** Small-M denoising steps of the action expert. */
object Action extends HartFactory {
  val component = "action"

  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = true)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new ActionHart(hartp, ThreadParam.ofHart(design, source), simp, traceIds)).hlink

}
