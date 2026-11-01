package examples.designs.piplus.thread

import framework.acc.balldomain.bbus.generated.BallComponents
import sims.configs.SimParam
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.configs.AccParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.acc.debug.AcceleratorDebugPort
import framework.acc.frontend.{Frontend, RoCCIO}
import framework.acc.memdomain.MemDomain
import framework.acc.balldomain.BallDomain
import framework.core.memory.MemCtrl
import framework.ip.bus.axi4

/** The accelerator of a piplus thread: Frontend, MemDomain and BallDomain with the piplus Balls; no GP domain. */
@instantiable
class Acc(
  accp:        AccParam,
  permissionp: PmpParams,
  trackingp:   TrackingParam,
  walkerp:     WalkerParam,
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
    val debug        = new AcceleratorDebugPort(frontp, memp, gpp, ballp, simp)
    val rocc         = new RoCCIO(frontp.commandDataBits, permissionp)
    val ddr          = new axi4.Port(transportp.axi)
    val readAddress  = Output(UInt(64.W))
    val writeAddress = Output(UInt(64.W))
    val memory       = new MemCtrl(trackingp, walkerp)
  })

  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontp, memp, gpp, ballp, permissionp, simp))

  val ballDomain: Instance[BallDomain] =
    Instantiate(new BallDomain(frontp, memp, gpp, ballp, permissionp, BallComponents.factories))

  val memDomain: Instance[MemDomain] =
    Instantiate(new MemDomain(frontp, memp, gpp, transportp, permissionp, trackingp, walkerp, simp))

  io.debug.frontend <> frontend.io.debug
  io.debug.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }
  io.debug.trace.memory        := memDomain.io.traceMemory
  io.debug.trace.ballStarted   := ballDomain.io.traceStarted
  io.debug.trace.ballCompleted := ballDomain.io.traceCompleted

  io.rocc <> frontend.io.rocc
  io.memory <> memDomain.io.memory
  io.ddr <> memDomain.io.ddr
  io.readAddress  := memDomain.io.readAddress
  io.writeAddress := memDomain.io.writeAddress

  memDomain.io.ball <> ballDomain.io.memory
  ballDomain.io.issue <> frontend.io.ballRequest
  frontend.io.ballCompletion <> ballDomain.io.complete

  memDomain.io.issue <> frontend.io.memoryRequest
  frontend.io.memoryCompletion <> memDomain.io.complete
  memDomain.io.mapping <> frontend.io.mappingUpdate

  memDomain.io.banks <> ballDomain.io.banks
  memDomain.io.ballLayoutRequest <> ballDomain.io.layoutRequest
  ballDomain.io.layoutResponse <> memDomain.io.ballLayoutResponse

  memDomain.io.shared.bank.mapping.ready        := true.B
  memDomain.io.shared.bank.config.ready         := true.B
  memDomain.io.shared.bank.local.request.valid  := false.B
  memDomain.io.shared.bank.local.request.bits   := 0.U.asTypeOf(memDomain.io.shared.bank.local.request.bits)
  memDomain.io.shared.bank.local.response.ready := true.B
  memDomain.io.shared.move.command.ready        := false.B
  memDomain.io.shared.move.completion.valid     := false.B
  memDomain.io.shared.move.completion.bits      := false.B
  frontend.io.barrier.release                   := false.B
  when(!reset.asBool) {
    assert(!memDomain.io.shared.move.command.valid, "A piplus thread cannot access another thread's banks")
    assert(!frontend.io.barrier.arrive, "A piplus thread cannot issue a core barrier")
  }
}
