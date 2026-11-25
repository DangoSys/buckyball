package examples.designs.poly.harts.poly

import framework.thread.configs.ThreadParam

import framework.thread.tbus.TBus

import examples.designs.poly.harts.poly.generated.ThreadComponents
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.execution.Execution
import framework.tlp.tss.Store
import framework.tlp.shm.{MeshSharedMemParams, MeshStorage, MeshThreadAttachment}
import framework.hart.memory.{Memory, Requester}
import framework.tlp.memory.{Dataflow, DmaMemory}
import framework.tlp.controller.Controller
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.generic.contract.sims.configs.SimParam

@instantiable
class PolyHart(
  cp:             HartParam,
  threads:        Seq[ThreadParam],
  simp:           SimParam,
  threadTraceIds: Seq[Int])
    extends Module
    with HasHLink {

  val diffTest = simp.diffTest
  require(
    cp.hlink.harts == 1 &&
      ((cp.hlink.backing && cp.hlink.requesters == 0) ||
        (!cp.hlink.backing && cp.hlink.requesters == 1)),
    "PolyHart requires one hart and a matching cache interface"
  )
  if (cp.threads.nonEmpty) {
    cp.threads.foreach { thread =>
      require(thread.link.dma == cp.interconnect.bus, "Thread DMA protocol must match the hart DDR interconnect")
    }
  }

  @public
  val hlink = IO(new HLinkIO(cp.hlink))

  val execution:  Instance[Execution]  =
    Instantiate(new Execution(cp.execution(directAccelerator = false), localRocc = false))
  val controller: Instance[Controller] = Instantiate(new Controller(cp.controller))
  val memory:     Instance[Memory]     =
    Instantiate(new Memory(cp.memory, externalHome = !cp.hlink.backing))

  execution.io.interrupts := hlink.interrupts(0)
  hlink.memory.backing.zip(memory.io.backing).foreach { case (port, backing) => port <> backing }

  val requester = Option.when(!cp.hlink.backing)(Instantiate(new Requester(
    cp.hlink.line,
    cp.hlink.requesterIds.head,
    Seq(cp.memory.icache.nodeId, cp.memory.dcache.cache.nodeId),
    2
  )))

  requester.foreach { node =>
    node.io.clients.take(2).zip(memory.io.coherent).foreach { case (port, cache) => port <> cache }
    hlink.memory.coherent(0) <> node.io.coherent
  }
  hlink.memory.uncached <> memory.io.uncached
  execution.io.workDrained := controller.io.drained
  controller.io.execution <> execution.io.controller.get
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := controller.io.memoryAllowed

  if (cp.threads.nonEmpty) {
    // Every thread owns a bank DMA channel to the DDR; the payload has the one after them.
    val transport = Instantiate(new DmaMemory(
      cp.interconnect.bus,
      cp.privileged.permission,
      cp.memory,
      cp.hlink.line.addressBits,
      cp.hlink.backing,
      cp.threads.size + 1
    ))
    val tbus: Instance[TBus] =
      Instantiate(new TBus(
        threads,
        cp.privileged.permission,
        cp.tss.get.memory,
        TBus.generators(
          threads,
          cp.privileged.permission,
          cp.hartId,
          simp,
          threadTraceIds,
          ThreadComponents.factories
        )
      ))
    require(cp.threads.size == 5)
    val storage  = cp.shm.get
    val geometry = MeshSharedMemParams(
      rows = 3,
      cols = 3,
      entriesPerBank = storage.entriesPerBank,
      dataBits = storage.dataBits,
      tagBits = 4,
      maxInFlight = 16,
      banksPerNode = storage.banks / 9,
      threads = Seq(
        MeshThreadAttachment(0, 2),
        MeshThreadAttachment(1, 2),
        MeshThreadAttachment(3, 2),
        MeshThreadAttachment(4, 2),
        MeshThreadAttachment(4, 2)
      )
    )
    val shm      = Instantiate(new MeshStorage(geometry, transfers = false))
    val dataflow = Instantiate(new Dataflow(geometry, cp.interconnect))
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
    dataflow.io.command <> controller.io.dataflow.get.request
    controller.io.dataflow.get.reply <> dataflow.io.reply
    controller.io.dataflow.get.busy   := dataflow.io.busy
    controller.io.stalled.get         := dataflow.io.stalled
    for (thread          <- cp.threads.indices) {
      dataflow.io.threads(thread) <> tbus.io.dma(thread)
      transport.io.clients(thread) <> dataflow.io.external(thread)
      transport.io.launch(thread).valid := controller.io.threads.get(thread).start.fire
      transport.io.launch(thread).bits  := controller.io.threads.get(thread).start.bits.context
      dataflow.io.task(thread).valid    := controller.io.threads.get(thread).start.fire
      dataflow.io.task(thread).bits     := controller.io.threads.get(thread).start.bits.descriptor.task
    }
    tbus.io.tss.foreach { port =>
      val tss: Instance[Store] = Instantiate(new Store(cp.tss.get))
      tss.io.clients <> port
      tss.io.host <> controller.io.tss.get.host
      tss.io.inUse                   := controller.io.tss.get.inUse
      tss.io.cancel                  := controller.io.tss.get.cancel
      controller.io.tss.get.busy     := tss.io.busy
      controller.io.tss.get.hostBusy := tss.io.hostBusy
    }
    val ddr = Instantiate(new framework.generic.bus.axi4.Interconnect(
      framework.generic.bus.axi4.InterconnectParam(cp.hlink.ddr, cp.threads.size + 1, None)
    ))
    for ((client, index) <- transport.io.axi.zipWithIndex) {
      val widen = Instantiate(new framework.generic.bus.axi4.Widen(cp.interconnect.bus, cp.hlink.ddr, 256))
      widen.io.in <> client
      ddr.io.in(index) <> widen.io.out
    }
    hlink.memory.ddr <> ddr.io.out
    val payload = Instantiate(new ServiceTransfer(threads.head.acc.get.transport, cp.privileged.permission))
    payload.io.host <> controller.io.remote.get
    hlink.service.get <> payload.io.remote
    hlink.payload.get <> payload.io.payload
    val context = Reg(new framework.generic.contract.tlink.TaskContext(cp.privileged.permission))
    val command = controller.io.execution.command
    when(command.fire) {
      context.satp               := command.bits.satp
      context.effectivePrivilege := command.bits.effectivePrivilege
      context.sum                := command.bits.sum
      context.mxr                := command.bits.mxr
      context.pmp                := command.bits.pmp
    }
    payload.io.context := context
    val client  = transport.io.clients(cp.threads.size)
    client.read <> payload.io.memory.read
    client.write <> payload.io.memory.write
    client.writeData <> payload.io.memory.writeData
    payload.io.memory.readData <> client.readData
    payload.io.memory.completed <> client.completed
    client.resolve.valid                := client.fault.valid
    client.resolve.bits                 := false.B
    transport.io.launch(cp.threads.size) <> payload.io.launch
    tbus.io.control.zip(controller.io.threads).foreach { case (port, control) => port <> control }
    controller.io.move.command.ready    := false.B
    controller.io.move.completion.valid := false.B
    controller.io.move.completion.bits  := false.B
    memory.io.dma.get <> transport.io.services
    hlink.memory.page.get <> transport.io.page
  } else {
    // Without threads the service moves no payload: no DMA asks about pages.
    hlink.memory.page.foreach(_.address := 0.U)
    hlink.service.zip(controller.io.remote).foreach { case (link, service) => link <> service }
    hlink.payload.get.send.valid        := false.B
    hlink.payload.get.send.bits         := 0.U
    hlink.payload.get.receive.ready     := false.B
    hlink.memory.ddr.aw.valid           := false.B
    hlink.memory.ddr.aw.bits            := 0.U.asTypeOf(hlink.memory.ddr.aw.bits)
    hlink.memory.ddr.w.valid            := false.B
    hlink.memory.ddr.w.bits             := 0.U.asTypeOf(hlink.memory.ddr.w.bits)
    hlink.memory.ddr.ar.valid           := false.B
    hlink.memory.ddr.ar.bits            := 0.U.asTypeOf(hlink.memory.ddr.ar.bits)
    hlink.memory.ddr.b.ready            := false.B
    hlink.memory.ddr.r.ready            := false.B
    controller.io.move.command.ready    := false.B
    controller.io.move.completion.valid := false.B
    controller.io.move.completion.bits  := false.B
  }
}

object Poly extends HartFactory {
  val component = "poly"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    require(source.getDataLinksCount == 1)
    val link = source.getDataLinks(0)
    require(link.getThreadsCount == source.getThreadIndicesCount &&
      (0 until link.getThreadsCount).forall(i => link.getThreads(i) == i))
    val p    = HartParam.fromProto(design, source, externalHome = true)
    require(link.getBanks == p.shm.get.banks, "Thread link capacity must match its hardware storage")
    require(ThreadParam.ofHart(design, source).forall(t => t.acc.get.memDomain.bankEntries == p.shm.get.entriesPerBank))
    p
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new PolyHart(hartp, ThreadParam.ofHart(design, source), simp, traceIds)).hlink

}
