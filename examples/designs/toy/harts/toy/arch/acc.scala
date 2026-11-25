package examples.designs.toy.harts.toy

import framework.generic.acc.configs.BallLink
import framework.generic.contract.ball.BallHostParam

import framework.generic.contract.memdomain.DmaFaultControl

import framework.ball.bbus.generated.BallComponents
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.rocket.configs.Params
import framework.hart.memory.{MemCtrl}
import framework.generic.bus.axi4
import framework.ball.bbus.BBus
import framework.custom.frontend.Frontend
import framework.custom.memdomain.MemDomain
import framework.custom.gpdomain.GpDomain
import framework.ball.BallDomain
import framework.custom.debug.Debug
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.frontend.RoCCIO

@instantiable
class Acc(config: Params, simp: SimParam) extends Module {
  val diffTest = simp.diffTest
  val hartp    = config
  val acc      = hartp.acc.get
  require(!acc.memDomain.sharedEnable)

  @public
  val io = IO(new Bundle {
    val rocc   = new RoCCIO(hartp.cpu.core.xLen, hartp.permission)
    val memory = new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker)
    val fault  = new DmaFaultControl
    val ddr    = new axi4.Port(acc.transport.axi)
  })

  val frontend: Instance[Frontend] =
    Instantiate(new Frontend(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, hartp.permission, simp))

  val memDomain: Instance[MemDomain] = Instantiate(new MemDomain(
    acc.frontend,
    acc.memDomain,
    acc.gpDomain,
    acc.transport,
    hartp.permission,
    hartp.memory.dcache.tracking,
    hartp.memory.memoryAccess.walker,
    simp,
    homeRanges = false
  ))

  val domain: Instance[BallDomain] =
    Instantiate(new BallDomain(
      BallLink.host(acc.frontend, acc.memDomain, acc.gpDomain),
      acc.ballDomain,
      hartp.permission,
      BallComponents.factories
    ))

  if (acc.gpDomain.enable) {
    val gp: Instance[GpDomain] = Instantiate(new GpDomain(acc.frontend, acc.memDomain, acc.gpDomain))
    gp.io.ownerId := hartp.hartId.U
    gp.io.command <> frontend.io.gpRequest.get
    frontend.io.gpCompletion.get <> gp.io.complete
    gp.io.image <> memDomain.io.image.get
    memDomain.io.gpBanks.get <> gp.io.banks
    memDomain.io.gpLayoutRequest.get <> gp.io.layoutRequest
    gp.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
    domain.io.operator.get.request <> gp.io.ball.request
    gp.io.ball.response <> domain.io.operator.get.response
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
  io.fault <> memDomain.io.fault

  val debug: Instance[Debug] = Instantiate(new Debug(
    acc.frontend,
    acc.memDomain,
    acc.gpDomain,
    acc.ballDomain,
    simp
  ))

  // The accelerator serves its hart.
  frontend.io.ownerId      := hartp.hartId.U
  memDomain.io.ownerId     := hartp.hartId.U
  domain.io.ownerId        := hartp.hartId.U
  debug.io.ids.trace       := hartp.hartId.U
  debug.io.ids.owner       := hartp.hartId.U
  debug.io.ids.sharedOwner := hartp.hartId.U

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
