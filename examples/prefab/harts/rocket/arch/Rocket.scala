package examples.hart.rocket

import framework.generic.acc.configs.BallLink
import framework.generic.contract.ball.BallHostParam

import framework.ball.bbus.BallFactory
import framework.hart.configs.HartParam
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import framework.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.rocket.configs.Params
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.memory.{MemorySystem}
import framework.custom.frontend.Frontend
import framework.custom.memdomain.MemDomain
import framework.custom.gpdomain.GpDomain
import framework.ball.BallDomain
import framework.ball.blink.HasBlink
import framework.custom.debug.Debug
import framework.generic.contract.sims.configs.SimParam

/** Rocket hart with local Ball endpoints. No task threads or TBus are instantiated. */
@instantiable
class Rocket(
  config:    Params,
  simp:      SimParam,
  factories: Map[String, BallFactory])
    extends Module
    with HasHLink {
  val diffTest = simp.diffTest
  require(config.acc.forall(a => !a.memDomain.sharedEnable))

  @public
  val hlink = IO(new HLinkIO(config.hlink))

  val execution: Instance[Execution]    = Instantiate(new Execution(config, config.acc.nonEmpty))
  val memory:    Instance[MemorySystem] = Instantiate(new MemorySystem(config.memory, !config.hlink.backing))
  execution.io.interrupts  := hlink.interrupts(0)
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.probe.get <> execution.io.probe
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  hlink.memory.uncached <> memory.io.uncached
  hlink.memory.backing.zip(memory.io.backing).foreach { case (port, source) => port <> source }
  hlink.memory.coherent <> memory.io.coherent
  memory.io.commandAllowed := true.B

  config.acc match {
    case Some(acc) =>
      val frontend:  Instance[Frontend]   =
        Instantiate(new Frontend(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, config.permission, simp))
      val memDomain: Instance[MemDomain]  = Instantiate(new MemDomain(
        acc.frontend,
        acc.memDomain,
        acc.gpDomain,
        acc.transport,
        config.permission,
        config.memory.dcache.tracking,
        config.memory.memoryAccess.walker,
        simp,
        homeRanges = false
      ))
      val domain:    Instance[BallDomain] =
        Instantiate(new BallDomain(
          BallLink.host(acc.frontend, acc.memDomain, acc.gpDomain),
          acc.ballDomain,
          config.permission,
          factories
        ))
      if (acc.gpDomain.enable) {
        val gp: Instance[GpDomain] = Instantiate(new GpDomain(acc.frontend, acc.memDomain, acc.gpDomain))
        gp.io.ownerId := config.hartId.U
        gp.io.command <> frontend.io.gpRequest.get
        frontend.io.gpCompletion.get <> gp.io.complete
        gp.io.image <> memDomain.io.image.get
        memDomain.io.gpBanks.get <> gp.io.banks
        memDomain.io.gpLayoutRequest.get <> gp.io.layoutRequest
        gp.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
        domain.io.operator.get.request <> gp.io.ball.request
        gp.io.ball.response <> domain.io.operator.get.response
      }
      frontend.io.rocc <> execution.io.rocc
      memory.io.commandAllowed := !frontend.io.rocc.busy
      domain.io.issue <> frontend.io.ballRequest
      frontend.io.ballCompletion <> domain.io.complete
      memDomain.io.ball <> domain.io.memory
      memDomain.io.issue <> frontend.io.memoryRequest
      frontend.io.memoryCompletion <> memDomain.io.complete
      memDomain.io.mapping <> frontend.io.mappingUpdate
      frontend.io.barrier.release := frontend.io.barrier.arrive
      memDomain.io.banks <> domain.io.banks
      memDomain.io.ballLayoutRequest <> domain.io.layoutRequest
      domain.io.layoutResponse <> memDomain.io.ballLayoutResponse
      hlink.memory.ddr <> memDomain.io.ddr
      memory.io.services <> memDomain.io.memory
      execution.io.fault <> memDomain.io.fault
      val debug: Instance[Debug] =
        Instantiate(new Debug(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, simp))
      // The accelerator serves its hart; its traces report as thread 0.
      frontend.io.ownerId                           := config.hartId.U
      memDomain.io.ownerId                          := config.hartId.U
      domain.io.ownerId                             := config.hartId.U
      debug.io.ids.trace                            := 0.U
      debug.io.ids.owner                            := config.hartId.U
      debug.io.ids.sharedOwner                      := 0.U
      debug.io.ant.valid                            := false.B
      debug.io.ant.bits                             := 0.U.asTypeOf(debug.io.ant.bits)
      debug.io.acc.frontend <> frontend.io.debug
      debug.io.acc.trace.memory                     := memDomain.io.traceMemory
      debug.io.acc.trace.ballStarted                := domain.io.traceStarted
      debug.io.acc.trace.ballCompleted              := domain.io.traceCompleted
      debug.io.acc.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }
      for (port <- memDomain.io.shared.bank.requests) {
        port.read.req.ready   := false.B
        port.read.resp.valid  := false.B
        port.read.resp.bits   := 0.U.asTypeOf(port.read.resp.bits)
        port.write.req.ready  := false.B
        port.write.resp.valid := false.B
        port.write.resp.bits  := 0.U.asTypeOf(port.write.resp.bits)
      }
      memDomain.io.shared.bank.config.ready         := false.B
      memDomain.io.shared.bank.mapping.ready        := false.B
      memDomain.io.shared.bank.local.request.valid  := false.B
      memDomain.io.shared.bank.local.request.bits   := 0.U.asTypeOf(memDomain.io.shared.bank.local.request.bits)
      memDomain.io.shared.bank.local.response.ready := false.B
      memDomain.io.shared.move.command.ready        := false.B
      memDomain.io.shared.move.completion.valid     := false.B
      memDomain.io.shared.move.completion.bits      := false.B
    case None      =>
      memory.io.services.cpuAllow          := true.B
      memory.io.services.maintenance.valid := false.B
      memory.io.services.maintenance.bits  := 0.U.asTypeOf(memory.io.services.maintenance.bits)
      memory.io.services.pteRequest.valid  := false.B
      memory.io.services.pteRequest.bits   := 0.U.asTypeOf(memory.io.services.pteRequest.bits)
      memory.io.services.maintained.ready  := true.B
      memory.io.services.pteResponse.ready := true.B
      memory.io.services.lineAddress       := 0.U
      memory.io.services.lineWrite         := false.B
      execution.io.rocc.cmd.ready          := false.B
      execution.io.rocc.resp.valid         := false.B
      execution.io.rocc.resp.bits          := 0.U.asTypeOf(execution.io.rocc.resp.bits)
      execution.io.rocc.busy               := false.B
      execution.io.rocc.interrupt          := false.B
      execution.io.fault.fault.valid       := false.B
      execution.io.fault.fault.bits        := 0.U.asTypeOf(execution.io.fault.fault.bits)
      hlink.memory.ddr.aw.valid            := false.B
      hlink.memory.ddr.aw.bits             := 0.U.asTypeOf(hlink.memory.ddr.aw.bits)
      hlink.memory.ddr.w.valid             := false.B
      hlink.memory.ddr.w.bits              := 0.U.asTypeOf(hlink.memory.ddr.w.bits)
      hlink.memory.ddr.ar.valid            := false.B
      hlink.memory.ddr.ar.bits             := 0.U.asTypeOf(hlink.memory.ddr.ar.bits)
      hlink.memory.ddr.b.ready             := true.B
      hlink.memory.ddr.r.ready             := true.B
  }
}

object RocketComponent extends HartFactory {
  val component = "rocket"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    val native = Params.fromProto(source); require(native.acc.isEmpty);
    HartParam.fromProto(source, Seq.empty).copy(hlink = native.hlink)
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new Rocket(Params.fromProto(source), simp, Map.empty)).hlink

}
