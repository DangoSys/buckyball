package examples.threads.attention

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
import framework.core.memory.MemCtrl
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

    val rocc         = new RoCCIO(commandDataBits, permissionp)
    val ddr          = new axi4.Port(transportp.axi)
    val readAddress  = Output(UInt(64.W))
    val writeAddress = Output(UInt(64.W))
    val memory       = new MemCtrl(trackingp, walkerp)

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

  require(!memp.sharedEnable && memp.sharedPorts == 0)

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
    assert(!memDomain.io.shared.move.command.valid, "Thread program cannot access another thread's banks")
    assert(!frontend.io.barrier.arrive, "Thread program cannot issue a core barrier")
  }

}
