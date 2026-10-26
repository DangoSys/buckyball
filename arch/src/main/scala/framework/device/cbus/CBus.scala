package framework.device.cbus

import sims.configs.SimParam
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.device.configs.DeviceParam
import framework.core.configs.CoreParam
import buckyball.config.{CorePlacement, Design}
import scala.jdk.CollectionConverters._
import framework.core.clink.{CLinkIO, CoreInterrupts}
import framework.device.ddr.Bridge
import framework.ip.bus.axi4
import framework.ip.bus.chi.RequesterPort
import framework.ip.bus.chi.snf.LineSramIO
import framework.device.dlink.BaseDevice

/** Routes core memory requests, shared-storage transfers and hardware interrupts. */
@instantiable
class CBus(b: DeviceParam, generators: Seq[() => CLinkIO]) extends BaseDevice(b.ddr.axi) {
  require(generators.size == b.cores.size)
  val corePorts     = generators.map(_())
  val ports         = corePorts
  require(
    corePorts.zip(b.cores).forall { case (port, corep) => port.p == corep.clink },
    "Core implementation must expose its configured CLink"
  )
  require(
    ports.forall(_.p.ddr == b.ddr.axi),
    "CBus DDR connections require matching AXI protocols or an explicit adapter"
  )
  require(
    ports.forall(_.p.uncached == b.ddr.axi),
    "CBus uncached connections require matching AXI protocols or an explicit adapter"
  )
  require(
    ports.filter(port => port.p.backing || port.p.requesters > 0).forall(_.p.line == b.ddr.line),
    "CBus line connections require matching CHI protocols or an explicit adapter"
  )
  val backingPorts  = corePorts.flatMap(_.memory.backing.toSeq)
  val coherentNodes = corePorts.flatMap(port => port.p.requesterIds.zip(port.memory.coherent))
  require(
    coherentNodes.map(_._1).sorted == (1 to coherentNodes.size),
    "CHI Home requires one connection for every requester node ID"
  )
  val coherentPorts = coherentNodes.sortBy(_._1).map(_._2)
  val bankPorts     = corePorts.flatMap(_.banks.toSeq)
  @public
  val banks         = IO(MixedVec(ports.flatMap(_.banks.toSeq).map(_.cloneType)))
  bankPorts.zip(banks).foreach { case (core, fabric) => fabric <> core }
  val hartCounts    = corePorts.map(_.interrupts.size)
  val hartBase      = hartCounts.scanLeft(0)(_ + _)

  @public
  val backing    = IO(Vec(backingPorts.size, Flipped(new LineSramIO(b.ddr.line))))
  @public
  val coherent   = IO(Vec(coherentPorts.size, new RequesterPort(b.ddr.line)))
  @public
  val memory     = IO(Vec(backingPorts.size, new LineSramIO(b.ddr.line)))
  @public
  val uncached   = IO(new axi4.Port(b.ddr.axi))
  @public
  val external   = IO(Flipped(new axi4.Port(b.ddr.axi)))
  @public
  val interrupts = IO(Input(Vec(hartCounts.sum, new CoreInterrupts)))

  for ((port, index) <- backingPorts.zipWithIndex) {
    backing(index).req <> port.request
    port.response <> backing(index).resp
  }
  for ((port, index) <- coherentPorts.zipWithIndex) coherent(index) <> port
  for {
    (core, index)    <- corePorts.zipWithIndex
    hart             <- core.interrupts.indices
  } {
    core.interrupts(hart) := interrupts(hartBase(index) + hart)
  }

  val linked = corePorts.indices.filter(i => corePorts(i).c2c.isDefined)
  if (linked.nonEmpty) {
    val fabric = Instantiate(new Fabric(b))
    for ((core, index) <- linked.zipWithIndex) {
      fabric.io.source(index) <> corePorts(core).c2c.get.tx
      corePorts(core).c2c.get.rx <> fabric.io.target(index)
    }
  }

  val bridgeMasters = if (backingPorts.nonEmpty) 1 else 0

  val memoryBus =
    Instantiate(new axi4.Interconnect(axi4.InterconnectParam(b.ddr.axi, corePorts.size + bridgeMasters + 1, None)))

  if (backingPorts.nonEmpty) {
    require(b.ddr.clients == backingPorts.size)
    val bridge = Instantiate(new Bridge(b.ddr))
    memoryBus.io.in(0) <> bridge.io.axi
    for (index <- backingPorts.indices) {
      bridge.io.request(index) <> memory(index).req
      memory(index).resp <> bridge.io.response(index)
    }
  }
  for ((core, index) <- corePorts.zipWithIndex) memoryBus.io.in(bridgeMasters + index) <> core.memory.ddr
  memoryBus.io.in(corePorts.size + bridgeMasters) <> external
  dlink.axi <> memoryBus.io.out

  val cpuMemory = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(b.ddr.axi, corePorts.size, None)))

  for ((core, index) <- corePorts.zipWithIndex) cpuMemory.io.in(index) <> core.memory.uncached
  uncached <> cpuMemory.io.out
}

trait CoreFactory {
  def component: String
  def parameters(design: Design, source: CorePlacement): CoreParam

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO

}

object CBus {

  def parameters(design: Design, components: Map[String, CoreFactory]): Seq[CoreParam] =
    design.getDevice.getCoresList.asScala.map { source =>
      components(source.getComponent).parameters(design, source)
    }.toSeq

  def generators(
    design:     Design,
    devicep:    DeviceParam,
    simp:       SimParam,
    components: Map[String, CoreFactory]
  ): Seq[() => CLinkIO] = {
    val sources = design.getDevice.getCoresList.asScala.toSeq
    require(sources.size == devicep.cores.size)
    sources.zip(devicep.cores).map { case (source, corep) =>
      require(source.getHartId == corep.hartId, "Core configuration order does not match design.pb")
      val factory  = components(source.getComponent)
      val traceIds = source.getThreadIndicesList.asScala.map(_.intValue + sources.size).toSeq
      () => factory.instantiate(source, corep, simp, traceIds)
    }
  }

}
