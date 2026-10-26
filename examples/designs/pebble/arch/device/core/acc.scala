package examples.designs.pebble

import framework.acc.balldomain.bbus.generated.BallComponents
import sims.configs.SimParam
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.memory.MemoryPort
import framework.acc.frontend.RoCCIO
import framework.acc.configs.AccParam
import framework.acc.frontend.Frontend
import framework.acc.memdomain.MemDomain
import framework.acc.debug.Debug
import framework.ip.bus.axi4
import framework.acc.balldomain.BallDomain
import framework.acc.balldomain.bbus.BBus

/** Pebble accelerator domains, concrete Balls and their local debug wiring. */
@instantiable
class Acc(corep: CoreParam, accp: AccParam, simp: SimParam) extends Module {
  val diffTest = simp.diffTest
  require(!accp.memDomain.sharedEnable && !accp.gpDomain.enable)

  @public
  val io = IO(new Bundle {
    val rocc   = new RoCCIO(corep.integer.xLen, corep.privileged.permission)
    val memory = new MemoryPort(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker)
    val ddr    = new axi4.Port(accp.transport.axi)
  })

  val frontend: Instance[Frontend] = Instantiate(new Frontend(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    corep.privileged.permission,
    simp
  ))

  val memDomain: Instance[MemDomain] = Instantiate(new MemDomain(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.transport,
    corep.privileged.permission,
    corep.memory.dcache.tracking,
    corep.memory.memoryAccess.walker,
    simp
  ))

  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    corep.privileged.permission,
    BallComponents.factories
  ))

  frontend.io.rocc <> io.rocc
  io.memory <> memDomain.io.memory

  // Local NPU domains are visible here;
  // BBUS owns concrete Ball instances.
  val debug: Instance[Debug] = Instantiate(new Debug(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    simp,
    traceId = corep.hartId,
    sharedOwnerId = corep.hartId
  ))

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
