package examples.designs.poly.harts.reduceunpack

import framework.generic.acc.configs.BallLink
import framework.generic.contract.ball.BallHostParam

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}

import framework.generic.acc.configs.AccParam
import framework.generic.contract.memdomain.DmaFaultControl
import framework.generic.acc.memdomain.PhysicalBankHash
import framework.ball.bbus.generated.BallComponents
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.custom.debug.AcceleratorDebugPort
import framework.custom.frontend.Frontend
import framework.custom.memdomain.MemDomain
import framework.custom.gpdomain.GpDomain
import framework.ball.BallDomain
import framework.ball.bbus.BBus
import framework.hart.memory.{MemCtrl}
import framework.tlp.shm.SharedLink
import framework.generic.bus.axi4
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.RoCCIO

@instantiable
class ReduceUnpackAcc(
  val accp:        AccParam,
  val permissionp: PmpParams,
  val trackingp:   TrackingParam,
  val walkerp:     WalkerParam,
  val simp:        SimParam,
  val homeRanges:  Boolean)
    extends Module {
  val commandDataBits = accp.frontend.commandDataBits
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
    val ownerId = Input(UInt(64.W))
    val debug   = new AcceleratorDebugPort(frontp, memp, gpp, ballp, simp)

    val rocc   = new RoCCIO(commandDataBits, permissionp)
    val ddr    = new axi4.Port(transportp.axi)
    val memory = new MemCtrl(trackingp, walkerp)
    val fault  = new DmaFaultControl
    val shm    = new SharedLink(frontp, memp)

    val shared_bank_hashes = Option.when(diffTest && memp.sharedEnable)(
      Input(Vec(sharedHashCount, new PhysicalBankHash(frontp)))
    )

  })

  // --- Instantiate domains ---
  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontp, memp, gpp, ballp, permissionp, simp))

  val ballDomain: Instance[BallDomain] =
    Instantiate(new BallDomain(
      BallLink.host(frontp, memp, gpp),
      ballp,
      permissionp,
      BallComponents.factories + ("mxint8unpack" -> examples.designs.poly.balls.mxint8unpack.Mxint8UnpackBall)
    ))

  val memDomain: Instance[MemDomain] =
    Instantiate(new MemDomain(
      frontp,
      memp,
      gpp,
      transportp,
      permissionp,
      trackingp,
      walkerp,
      simp,
      homeRanges = homeRanges
    ))

  val gpDomain =
    if (gpp.enable) Some(Instantiate(new GpDomain(frontp, memp, gpp)))
    else None

  memDomain.io.shared_bank_hashes.foreach(_ := io.shared_bank_hashes.get)

  io.debug.frontend <> frontend.io.debug
  frontend.io.ownerId           := io.ownerId
  ballDomain.io.ownerId         := io.ownerId
  memDomain.io.ownerId          := io.ownerId
  gpDomain.foreach(_.io.ownerId := io.ownerId)
  io.debug.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, source) => sink := source }
  io.debug.trace.memory         := memDomain.io.traceMemory
  io.debug.trace.ballStarted    := ballDomain.io.traceStarted
  io.debug.trace.ballCompleted  := ballDomain.io.traceCompleted

  io.rocc <> frontend.io.rocc
  io.memory <> memDomain.io.memory
  io.fault <> memDomain.io.fault
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
    ballDomain.io.operator.get <> controller.io.ball
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
