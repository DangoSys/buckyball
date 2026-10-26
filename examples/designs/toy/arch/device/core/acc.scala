package examples.designs.toy

import framework.acc.balldomain.bbus.generated.BallComponents
import sims.configs.SimParam
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.rocket.configs.Params
import framework.core.memory.MemoryPort
import framework.acc.frontend.RoCCIO
import framework.ip.bus.axi4
import framework.acc.balldomain.bbus.BBus
import framework.acc.frontend.Frontend
import framework.acc.memdomain.MemDomain
import framework.acc.gpdomain.GpDomain
import framework.acc.balldomain.BallDomain
import framework.acc.debug.Debug

@instantiable
class Acc(config: Params, simp: SimParam) extends Module {
  val diffTest = simp.diffTest
  val corep    = config
  val acc      = corep.acc.get
  require(!acc.memDomain.sharedEnable)

  @public
  val io = IO(new Bundle {
    val rocc   = new RoCCIO(corep.cpu.core.xLen, corep.permission)
    val memory = new MemoryPort(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker)
    val ddr    = new axi4.Port(acc.transport.axi)
  })

  val frontend: Instance[Frontend] =
    Instantiate(new Frontend(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, corep.permission, simp))

  val memDomain: Instance[MemDomain] = Instantiate(new MemDomain(
    acc.frontend,
    acc.memDomain,
    acc.gpDomain,
    acc.transport,
    corep.permission,
    corep.memory.dcache.tracking,
    corep.memory.memoryAccess.walker,
    simp
  ))

  val domain: Instance[BallDomain] =
    Instantiate(new BallDomain(
      acc.frontend,
      acc.memDomain,
      acc.gpDomain,
      acc.ballDomain,
      corep.permission,
      BallComponents.factories
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
  frontend.io.rocc <> io.rocc
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
  io.ddr <> memDomain.io.ddr
  io.memory <> memDomain.io.memory

  val debug: Instance[Debug] = Instantiate(new Debug(
    acc.frontend,
    acc.memDomain,
    acc.gpDomain,
    acc.ballDomain,
    simp,
    traceId = corep.hartId,
    sharedOwnerId = corep.hartId
  ))

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
}
