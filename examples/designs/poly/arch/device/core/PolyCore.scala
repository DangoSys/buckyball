package examples.designs.poly

import framework.core.tbus.generated.ThreadComponents
import sims.configs.SimParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.tbus.TBus
import framework.ip.bus.chi.rnf.AxiCache
import framework.core.execution.Execution
import framework.core.memory.Memory
import framework.core.tss.Store
import framework.core.shm.SHM
import framework.core.controller.Controller
import framework.core.clink.{CLinkIO, HasCLink}

@instantiable
class PolyCore(cp: CoreParam, simp: SimParam, threadTraceIds: Seq[Int]) extends Module with HasCLink {

  val diffTest = simp.diffTest
  require(
    cp.clink.harts == 1 &&
      ((cp.clink.backing && cp.clink.requesters == 0) ||
        (!cp.clink.backing && cp.clink.requesters == 2 + cp.dmaCache.size)),
    "PolyCore requires one hart and a matching cache interface"
  )
  if (!cp.clink.backing) {
    require(
      cp.dmaCache.nonEmpty == cp.threads.nonEmpty,
      "Thread DMA requires its own coherent cache requester"
    )
  }
  if (cp.threads.nonEmpty) {
    if (cp.clink.shared)
      require(cp.clink.c2c.contains(cp.c2c.axi), "C2C protocol must match its CLink channel")
    val tracking = cp.memory.dcache.tracking
    cp.threads.foreach { thread =>
      val transport = thread.acc.get.transport
      require(transport.axi == cp.interconnect.bus, "Thread DMA protocol must match the core DDR interconnect")
      require(
        transport.prepared.bus.addressBits == cp.memory.memoryAccess.memory.chi.addressBits,
        "Thread page-table address width must match the core memory service"
      )
      require(
        (
          transport.interlock.addressBits,
          transport.interlock.idBits,
          transport.interlock.lineBytes,
          transport.interlock.maxRanges
        ) ==
          (tracking.addressBits, tracking.idBits, tracking.lineBytes, tracking.maxRanges),
        "Thread cache-maintenance interface must match the core memory service"
      )
    }
  }

  @public
  val clink = IO(new CLinkIO(cp.clink))

  val execution:  Instance[Execution]  =
    Instantiate(new Execution(cp.execution(directAccelerator = false), localRocc = false))
  val controller: Instance[Controller] = Instantiate(new Controller(cp.controller))
  val memory:     Instance[Memory]     =
    Instantiate(new Memory(cp.memory, externalHome = !cp.clink.backing))

  execution.io.interrupts  := clink.interrupts(0)
  clink.memory.backing.zip(memory.io.backing).foreach { case (port, backing) => port <> backing }
  clink.memory.coherent.take(2).zip(memory.io.coherent).foreach { case (port, cache) => port <> cache }
  clink.memory.uncached <> memory.io.uncached
  execution.io.workDrained := controller.io.drained
  controller.io.execution <> execution.io.controller.get
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := controller.io.memoryAllowed

  if (cp.threads.nonEmpty) {
    val tbus: Instance[TBus] =
      Instantiate(new TBus(
        cp,
        simp,
        threadTraceIds,
        TBus.generators(cp, simp, threadTraceIds, ThreadComponents.factories)
      ))
    tbus.io.shm.foreach { port =>
      val shm: Instance[SHM] = Instantiate(new SHM(cp.sharedFrontend, cp.sharedMemory, cp.tbus.get))
      shm.io <> port
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
    if (cp.clink.backing) {
      require(cp.interconnect.bus == cp.clink.ddr, "Thread DDR interconnect must match CLink")
      clink.memory.ddr <> tbus.io.ddr
    } else {
      val dma = Instantiate(new AxiCache(cp.interconnect.bus, cp.dmaCache.get))
      dma.io.axi <> tbus.io.ddr
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
    clink.c2c.zip(tbus.io.c2c).foreach { case (link, port) => link <> port }
    tbus.io.control.zip(controller.io.threads).foreach { case (port, control) => port <> control }
    tbus.io.remote.zip(controller.io.remote).foreach { case (port, remote) => port <> remote }
    tbus.io.move <> controller.io.move
    tbus.io.memory.zip(memory.io.threads).foreach { case (port, service) => port <> service }
  } else {
    require(!cp.clink.shared, "A core without a shared bank network must not advertise a C2C endpoint")
    clink.memory.ddr.aw.valid           := false.B
    clink.memory.ddr.aw.bits            := 0.U.asTypeOf(clink.memory.ddr.aw.bits)
    clink.memory.ddr.w.valid            := false.B
    clink.memory.ddr.w.bits             := 0.U.asTypeOf(clink.memory.ddr.w.bits)
    clink.memory.ddr.ar.valid           := false.B
    clink.memory.ddr.ar.bits            := 0.U.asTypeOf(clink.memory.ddr.ar.bits)
    clink.memory.ddr.b.ready            := false.B
    clink.memory.ddr.r.ready            := false.B
    controller.io.move.command.ready    := false.B
    controller.io.move.completion.valid := false.B
    controller.io.move.completion.bits  := false.B
  }
}

object Poly extends CoreFactory {
  val component = "poly"
  def parameters(design: Design, source: CorePlacement): CoreParam =
    CoreParam.fromProto(design, source, externalHome = true)

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new PolyCore(corep, simp, traceIds)).clink

}
