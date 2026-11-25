package framework.hart.cbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.configs.HartParam
import buckyball.config.{Design, HartPlacement}
import scala.jdk.CollectionConverters._
import framework.hart.hlink.HLinkIO
import framework.generic.ddr.{Bridge, Params => DdrParam}
import framework.generic.bus.axi4
import framework.generic.bus.chi.RequesterPort
import framework.generic.memory.coherence.PageQuery
import framework.generic.bus.chi.snf.LineSramIO
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.hlink.HartInterrupts

/** The memory port the harts of the bus share. */
class MemoryLink(p: axi4.Params) extends Bundle {
  val axi = new axi4.Port(p)
}

/** Routes hart memory requests, management services and hardware interrupts. */
@instantiable
class CBus(harts: Seq[HartParam], ddr: DdrParam, generators: Seq[() => HLinkIO]) extends Module {
  require(generators.size == harts.size)

  @public
  val dlink         = IO(new MemoryLink(ddr.port))
  val hartPorts     = generators.map(_())
  val ports         = hartPorts
  require(
    hartPorts.zip(harts).forall { case (port, hartp) => port.p == hartp.hlink },
    "Hart implementation must expose its configured HLink"
  )
  require(
    ports.forall(_.p.ddr == ddr.port),
    "CBus DDR connections require matching AXI protocols or an explicit adapter"
  )
  require(
    ports.forall(_.p.uncached == ddr.axi),
    "CBus uncached connections require matching AXI protocols or an explicit adapter"
  )
  require(
    ports.filter(port => port.p.backing || port.p.requesters > 0).forall(_.p.line == ddr.line),
    "CBus line connections require matching CHI protocols or an explicit adapter"
  )
  val backingPorts  = hartPorts.flatMap(_.memory.backing.toSeq)
  val coherentNodes = hartPorts.flatMap(port => port.p.requesterIds.zip(port.memory.coherent))
  require(
    coherentNodes.map(_._1).sorted == (1 to coherentNodes.size),
    "CHI Home requires one connection for every requester node ID"
  )
  val coherentPorts = coherentNodes.sortBy(_._1).map(_._2)
  val pagePorts     = hartPorts.flatMap(_.memory.page.toSeq)
  val hartCounts    = hartPorts.map(_.interrupts.size)
  val hartBase      = hartCounts.scanLeft(0)(_ + _)

  @public
  val backing    = IO(Vec(backingPorts.size, Flipped(new LineSramIO(ddr.line))))
  @public
  val coherent   = IO(Vec(coherentPorts.size, new RequesterPort(ddr.line)))
  @public
  val pages      = IO(Vec(pagePorts.size, new PageQuery(ddr.line.addressBits)))
  @public
  val memory     = IO(Vec(backingPorts.size, new LineSramIO(ddr.line)))
  @public
  val uncached   = IO(new axi4.Port(ddr.axi))
  @public
  val external   = IO(Flipped(new axi4.Port(ddr.axi)))
  @public
  val interrupts = IO(Input(Vec(hartCounts.sum, new HartInterrupts)))

  for ((port, index) <- backingPorts.zipWithIndex) {
    backing(index).req <> port.request
    port.response <> backing(index).resp
  }
  for ((port, index) <- coherentPorts.zipWithIndex) coherent(index) <> port
  for ((port, index) <- pagePorts.zipWithIndex) pages(index) <> port
  for {
    (hart, index)    <- hartPorts.zipWithIndex
    line             <- hart.interrupts.indices
  } {
    hart.interrupts(line) := interrupts(hartBase(index) + line)
  }

  val serviceEndpoints = hartPorts.flatMap(_.service.toSeq)
  @public
  val services         = IO(MixedVec(serviceEndpoints.map(_.cloneType)))
  serviceEndpoints.zip(services).foreach { case (hart, service) => service <> hart }

  val payloadEndpoints = hartPorts.flatMap(_.payload.toSeq)
  @public
  val payloads         = IO(MixedVec(payloadEndpoints.map(_.cloneType)))
  payloadEndpoints.zip(payloads).foreach { case (hart, payload) => payload <> hart }

  val bridgeMasters    = if (backingPorts.nonEmpty) 1 else 0

  val memoryBus =
    Instantiate(new axi4.Interconnect(axi4.InterconnectParam(ddr.port, hartPorts.size + bridgeMasters + 1, None)))

  if (backingPorts.nonEmpty) {
    require(ddr.clients == backingPorts.size)
    val bridge = Instantiate(new Bridge(ddr))
    memoryBus.io.in(0) <> bridge.io.axi
    for (index <- backingPorts.indices) {
      bridge.io.request(index) <> memory(index).req
      memory(index).resp <> bridge.io.response(index)
    }
  }
  for ((hart, index) <- hartPorts.zipWithIndex) memoryBus.io.in(bridgeMasters + index) <> hart.memory.ddr
  // The CPUs' uncached DDR accesses, a bank row a beat, on the DDR's beat.
  val widen = Instantiate(new axi4.Widen(ddr.axi, ddr.port, 1))
  widen.io.in <> external
  memoryBus.io.in(hartPorts.size + bridgeMasters) <> widen.io.out
  dlink.axi <> memoryBus.io.out

  val cpuMemory = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(ddr.axi, hartPorts.size, None)))

  for ((hart, index) <- hartPorts.zipWithIndex) cpuMemory.io.in(index) <> hart.memory.uncached
  uncached <> cpuMemory.io.out
}

trait HartFactory {
  def component: String
  def parameters(design: Design, source: HartPlacement): HartParam

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO

}

object CBus {

  def parameters(design: Design, components: Map[String, HartFactory]): Seq[HartParam] =
    design.getDevice.getHartsList.asScala.map { source =>
      components(source.getComponent).parameters(design, source)
    }.toSeq

  def generators(
    design:     Design,
    harts:      Seq[HartParam],
    simp:       SimParam,
    components: Map[String, HartFactory]
  ): Seq[() => HLinkIO] = {
    val sources = design.getDevice.getHartsList.asScala.toSeq
    require(sources.size == harts.size)
    sources.zip(harts).map { case (source, hartp) =>
      require(source.getHartId == hartp.hartId, "Hart configuration order does not match design.pb")
      val factory  = components(source.getComponent)
      val traceIds = source.getThreadIndicesList.asScala.map(_.intValue + sources.size).toSeq
      () => factory.instantiate(design, source, hartp, simp, traceIds)
    }
  }

}
