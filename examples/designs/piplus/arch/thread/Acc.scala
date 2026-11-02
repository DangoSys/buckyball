package examples.designs.piplus.thread

import framework.acc.balldomain.bbus.generated.BallComponents
import sims.configs.SimParam
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.configs.AccParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.debug.AcceleratorDebugPort
import framework.acc.frontend.{Frontend, RoCCIO}
import framework.acc.memdomain.ThreadMemDomain
import framework.acc.balldomain.BallDomain
import framework.ip.bus.axi4

/** The accelerator of a piplus thread: Frontend, MemDomain and BallDomain with the piplus Balls; no GP domain. */
@instantiable
class Acc(
  accp:        AccParam,
  permissionp: PmpParams,
  simp:        SimParam)
    extends Module {
  val frontp     = accp.frontend
  val memp       = accp.memDomain
  val gpp        = accp.gpDomain
  val ballp      = accp.ballDomain
  val transportp = accp.transport

  require(!gpp.enable, "piplus threads have no GP domain")
  require(!memp.sharedEnable && memp.sharedPorts == 0, "piplus threads have no shared banks")

  @public
  val io = IO(new Bundle {
    val debug = new AcceleratorDebugPort(frontp, memp, gpp, ballp, simp)
    val rocc  = new RoCCIO(frontp.commandDataBits, permissionp)
    val dma   = new framework.thread.tlink.Dma(transportp.axi)
  })

  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontp, memp, gpp, ballp, permissionp, simp))

  val ballDomain: Instance[BallDomain] =
    Instantiate(new BallDomain(frontp, memp, gpp, ballp, permissionp, BallComponents.factories))

  val memDomain: Instance[ThreadMemDomain] =
    Instantiate(new ThreadMemDomain(frontp, memp, gpp, transportp, permissionp, simp))

  io.debug.frontend <> frontend.io.debug
  io.debug.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }
  io.debug.trace.memory        := memDomain.io.traceMemory
  io.debug.trace.ballStarted   := ballDomain.io.traceStarted
  io.debug.trace.ballCompleted := ballDomain.io.traceCompleted

  io.rocc <> frontend.io.rocc
  io.dma <> memDomain.io.dma

  memDomain.io.ball <> ballDomain.io.memory
  ballDomain.io.issue <> frontend.io.ballRequest
  frontend.io.ballCompletion <> ballDomain.io.complete

  memDomain.io.issue <> frontend.io.memoryRequest
  frontend.io.memoryCompletion <> memDomain.io.complete
  memDomain.io.mapping <> frontend.io.mappingUpdate

  memDomain.io.banks <> ballDomain.io.banks
  memDomain.io.ballLayoutRequest <> ballDomain.io.layoutRequest
  ballDomain.io.layoutResponse <> memDomain.io.ballLayoutResponse

  frontend.io.barrier.release := false.B
  when(!reset.asBool) {
    assert(!frontend.io.barrier.arrive, "A piplus thread cannot issue a core barrier")
  }
}
