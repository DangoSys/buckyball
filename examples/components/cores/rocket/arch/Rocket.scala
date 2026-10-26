package examples.core.rocket

import framework.acc.balldomain.bbus.BallFactory
import sims.configs.SimParam
import framework.core.configs.CoreParam
import framework.device.cbus.CoreFactory
import buckyball.config.{CorePlacement, Design}

import framework.acc.balldomain.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.rocket.configs.Params
import framework.core.clink.{CLinkIO, HasCLink}
import framework.core.memory.MemorySystem
import framework.acc.frontend.Frontend
import framework.acc.memdomain.MemDomain
import framework.acc.gpdomain.GpDomain
import framework.acc.balldomain.BallDomain
import framework.acc.balldomain.ball.blink.HasBlink
import framework.acc.debug.Debug

/** Rocket core with local Ball endpoints. No task threads or TBus are instantiated. */
@instantiable
class Rocket(
  config:    Params,
  simp:      SimParam,
  factories: Map[String, BallFactory])
    extends Module
    with HasCLink {
  val diffTest = simp.diffTest
  require(config.acc.forall(a => !a.memDomain.sharedEnable))

  @public
  val clink = IO(new CLinkIO(config.clink))

  val execution: Instance[Execution]    = Instantiate(new Execution(config, config.acc.nonEmpty))
  val memory:    Instance[MemorySystem] = Instantiate(new MemorySystem(config.memory, !config.clink.backing))
  execution.io.interrupts  := clink.interrupts(0)
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  clink.memory.uncached <> memory.io.uncached
  clink.memory.backing.zip(memory.io.backing).foreach { case (port, source) => port <> source }
  clink.memory.coherent <> memory.io.coherent
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
        simp
      ))
      val domain:    Instance[BallDomain] =
        Instantiate(new BallDomain(
          acc.frontend,
          acc.memDomain,
          acc.gpDomain,
          acc.ballDomain,
          config.permission,
          factories
        ))
      if (acc.gpDomain.enable) {
        val gp: Instance[GpDomain] = Instantiate(new GpDomain(acc.frontend, acc.memDomain, acc.gpDomain))
        gp.io.command <> frontend.io.gpRequest.get
        frontend.io.gpCompletion.get <> gp.io.complete
        gp.io.image <> memDomain.io.image.get
        memDomain.io.gpBanks.get <> gp.io.banks
        memDomain.io.gpLayoutRequest.get <> gp.io.layoutRequest
        gp.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
        domain.io.kernel.get.request <> gp.io.ball.request
        gp.io.ball.response <> domain.io.kernel.get.response
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
      clink.memory.ddr <> memDomain.io.ddr
      memory.io.services <> memDomain.io.memory
      val debug: Instance[Debug] =
        Instantiate(new Debug(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, simp, 0, 0))
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
      execution.io.rocc.cmd.ready          := false.B
      execution.io.rocc.resp.valid         := false.B
      execution.io.rocc.resp.bits          := 0.U.asTypeOf(execution.io.rocc.resp.bits)
      execution.io.rocc.busy               := false.B
      execution.io.rocc.interrupt          := false.B
      clink.memory.ddr.aw.valid            := false.B
      clink.memory.ddr.aw.bits             := 0.U.asTypeOf(clink.memory.ddr.aw.bits)
      clink.memory.ddr.w.valid             := false.B
      clink.memory.ddr.w.bits              := 0.U.asTypeOf(clink.memory.ddr.w.bits)
      clink.memory.ddr.ar.valid            := false.B
      clink.memory.ddr.ar.bits             := 0.U.asTypeOf(clink.memory.ddr.ar.bits)
      clink.memory.ddr.b.ready             := true.B
      clink.memory.ddr.r.ready             := true.B
  }
}

object RocketComponent extends CoreFactory {
  val component = "rocket"

  def parameters(design: Design, source: CorePlacement): CoreParam = {
    val native = Params.fromProto(source); require(native.acc.isEmpty);
    CoreParam.fromProto(source, Seq.empty).copy(clink = native.clink)
  }

  def instantiate(
    source:   CorePlacement,
    corep:    CoreParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): CLinkIO =
    Instantiate(new Rocket(Params.fromProto(source), simp, Map.empty)).clink

}
