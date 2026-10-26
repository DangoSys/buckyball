package examples.threads.matmul

import framework.acc.balldomain.bbus.generated.BallComponents
import sims.configs.SimParam
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.configs.AccParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.acc.debug.AcceleratorDebugPort
import framework.acc.frontend.RoCCIO
import framework.acc.frontend.Frontend
import framework.acc.memdomain.{MemDomain, PhysicalBankHash}
import framework.acc.gpdomain.GpDomain
import framework.acc.balldomain.BallDomain
import framework.acc.balldomain.bbus.BBus
import framework.core.memory.MemoryPort
import framework.thread.tlink.SharedLink
import framework.ip.bus.axi4

@instantiable
class Acc(
  val accp:        AccParam,
  val permissionp: PmpParams,
  val trackingp:   TrackingParam,
  val walkerp:     WalkerParam,
  val simp:        SimParam)
    extends Module {
  val commandDataBits = accp.frontend.commandDataBits
  val ownerId         = accp.frontend.ownerId
  val sharedPorts     = accp.memDomain.sharedPorts
  val diffTest        = simp.diffTest
  val frontp          = accp.frontend
  val memp            = accp.memDomain
  val gpp             = accp.gpDomain
  val ballp           = accp.ballDomain
  val transportp      = accp.transport

  val sharedHashCount = if (memp.sharedEnable) memp.sharedBankNum else 0

  @public
  val io = IO(new Bundle {
    val debug = new AcceleratorDebugPort(frontp, memp, gpp, ballp, simp)

    val rocc   = new RoCCIO(commandDataBits, permissionp)
    val ddr    = new axi4.Port(transportp.axi)
    val memory = new MemoryPort(trackingp, walkerp)
    val shm    = new SharedLink(frontp, memp)

    val shared_bank_hashes = Option.when(diffTest && memp.sharedEnable)(
      Input(Vec(sharedHashCount, new PhysicalBankHash(frontp)))
    )

  })

  // --- Instantiate domains ---
  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontp, memp, gpp, ballp, permissionp, simp))

  val ballDomain: Instance[BallDomain] =
    Instantiate(new BallDomain(frontp, memp, gpp, ballp, permissionp, BallComponents.factories))

  val memDomain: Instance[MemDomain] =
    Instantiate(new MemDomain(frontp, memp, gpp, transportp, permissionp, trackingp, walkerp, simp))

  val gpDomain =
    if (gpp.enable) Some(Instantiate(new GpDomain(frontp, memp, gpp)))
    else None

  memDomain.io.shared_bank_hashes.foreach(_ := io.shared_bank_hashes.get)

  io.debug.frontend <> frontend.io.debug
  io.debug.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }
  io.debug.trace.memory        := memDomain.io.traceMemory
  io.debug.trace.ballStarted   := ballDomain.io.traceStarted
  io.debug.trace.ballCompleted := ballDomain.io.traceCompleted

  io.rocc <> frontend.io.rocc
  io.memory <> memDomain.io.memory
  io.ddr <> memDomain.io.ddr

  // --- Frontend -> BallDomain ---
  memDomain.io.ball <> ballDomain.io.memory
  ballDomain.io.issue <> frontend.io.ballRequest
  frontend.io.ballCompletion <> ballDomain.io.complete

  // --- Frontend -> MemDomain ---
  memDomain.io.issue <> frontend.io.memoryRequest
  frontend.io.memoryCompletion <> memDomain.io.complete
  memDomain.io.mapping <> frontend.io.mappingUpdate

  if (gpp.enable) {
    val controller = gpDomain.get
    controller.io.command <> frontend.io.gpRequest.get
    frontend.io.gpCompletion.get <> controller.io.complete
    controller.io.image <> memDomain.io.image.get
    memDomain.io.gpBanks.get <> controller.io.banks
    memDomain.io.gpLayoutRequest.get <> controller.io.layoutRequest
    controller.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
    ballDomain.io.kernel.get <> controller.io.ball
  }

  memDomain.io.banks <> ballDomain.io.banks
  memDomain.io.ballLayoutRequest <> ballDomain.io.layoutRequest
  ballDomain.io.layoutResponse <> memDomain.io.ballLayoutResponse

  // --- Shared memory passthrough ---
  io.shm.bank.requests <> memDomain.io.shared.bank.requests
  io.shm.move <> memDomain.io.shared.move
  io.shm.bank.local <> memDomain.io.shared.bank.local
  io.shm.bank.config <> memDomain.io.shared.bank.config
  io.shm.bank.mapping <> memDomain.io.shared.bank.mapping

  // --- Barrier passthrough ---
  io.shm.barrierArrive        := frontend.io.barrier.arrive
  frontend.io.barrier.release := io.shm.barrierRelease

}
