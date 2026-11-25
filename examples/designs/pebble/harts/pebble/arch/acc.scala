package examples.designs.pebble.harts.pebble

import framework.generic.acc.configs.BallLink
import framework.generic.contract.ball.BallHostParam

import framework.generic.acc.configs.AccParam
import framework.generic.contract.memdomain.DmaFaultControl

import framework.ball.bbus.generated.BallComponents
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.memory.{MemCtrl}
import framework.custom.frontend.Frontend
import framework.custom.memdomain.MemDomain
import framework.custom.debug.Debug
import framework.generic.bus.axi4
import framework.ball.BallDomain
import framework.ball.bbus.BBus
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.frontend.RoCCIO

/** Pebble accelerator domains, concrete Balls and their local debug wiring. */
@instantiable
class Acc(hartp: HartParam, accp: AccParam, simp: SimParam) extends Module {
  val diffTest = simp.diffTest
  require(!accp.memDomain.sharedEnable && !accp.gpDomain.enable)

  @public
  val io = IO(new Bundle {
    val rocc   = new RoCCIO(hartp.integer.xLen, hartp.privileged.permission)
    val memory = new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker)
    val fault  = new DmaFaultControl
    val ddr    = new axi4.Port(accp.transport.axi)
  })

  val frontend: Instance[Frontend] = Instantiate(new Frontend(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    hartp.privileged.permission,
    simp
  ))

  val memDomain: Instance[MemDomain] = Instantiate(new MemDomain(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.transport,
    hartp.privileged.permission,
    hartp.memory.dcache.tracking,
    hartp.memory.memoryAccess.walker,
    simp,
    homeRanges = false
  ))

  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(
    BallLink.host(accp.frontend, accp.memDomain, accp.gpDomain),
    accp.ballDomain,
    hartp.privileged.permission,
    BallComponents.factories
  ))

  frontend.io.rocc <> io.rocc
  io.memory <> memDomain.io.memory
  io.fault <> memDomain.io.fault

  // Local NPU domains are visible here;
  // BBUS owns concrete Ball instances.
  val debug: Instance[Debug] = Instantiate(new Debug(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    simp
  ))

  // The accelerator serves its hart.
  frontend.io.ownerId      := hartp.hartId.U
  memDomain.io.ownerId     := hartp.hartId.U
  ballDomain.io.ownerId    := hartp.hartId.U
  debug.io.ids.trace       := hartp.hartId.U
  debug.io.ids.owner       := hartp.hartId.U
  debug.io.ids.sharedOwner := hartp.hartId.U

  debug.io.ant.valid               := false.B
  debug.io.ant.bits                := 0.U.asTypeOf(debug.io.ant.bits)
  debug.io.acc.frontend <> frontend.io.debug
  debug.io.acc.trace.memory        := memDomain.io.traceMemory
  debug.io.acc.trace.ballStarted   := ballDomain.io.traceStarted
  debug.io.acc.trace.ballCompleted := ballDomain.io.traceCompleted
  debug.io.acc.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }

  ballDomain.io.issue <> frontend.io.ballRequest
  frontend.io.ballCompletion <> ballDomain.io.complete
  memDomain.io.ball <> ballDomain.io.memory
  memDomain.io.issue <> frontend.io.memoryRequest
  frontend.io.memoryCompletion <> memDomain.io.complete
  memDomain.io.banks <> ballDomain.io.banks
  memDomain.io.ballLayoutRequest <> ballDomain.io.layoutRequest
  ballDomain.io.layoutResponse <> memDomain.io.ballLayoutResponse
  memDomain.io.mapping <> frontend.io.mappingUpdate
  frontend.io.barrier.release      := frontend.io.barrier.arrive
  io.ddr <> memDomain.io.ddr

  // Pebble has no shared-memory connection;
  // its complete data path is local.
  for (port <- memDomain.io.shared.bank.requests) {
    port.read.req.ready   := false.B
    port.read.resp.valid  := false.B
    port.read.resp.bits   := 0.U.asTypeOf(port.read.resp.bits)
    port.write.req.ready  := false.B
    port.write.resp.valid := false.B
    port.write.resp.bits  := 0.U.asTypeOf(port.write.resp.bits)
  }
  memDomain.io.shared.bank.config.ready := false.B
  memDomain.io.shared.bank.mapping.ready        := false.B
  memDomain.io.shared.bank.local.request.valid  := false.B
  memDomain.io.shared.bank.local.request.bits   := 0.U.asTypeOf(memDomain.io.shared.bank.local.request.bits)
  memDomain.io.shared.bank.local.response.ready := false.B
  memDomain.io.shared.move.command.ready        := false.B
  memDomain.io.shared.move.completion.valid     := false.B
  memDomain.io.shared.move.completion.bits      := false.B
}
