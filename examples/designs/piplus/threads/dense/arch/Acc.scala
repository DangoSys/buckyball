package examples.designs.piplus.threads.dense

import framework.thread.acc.configs.BallLink

import framework.generic.contract.ball.BallHostParam

import framework.thread.acc.configs.AccParam

import framework.ball.bbus.generated.BallComponents
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.acc.debug.AcceleratorDebugPort
import framework.thread.acc.frontend.Frontend
import framework.thread.acc.memdomain.ThreadMemDomain
import framework.thread.acc.gpdomain.GpDomain
import framework.ball.BallDomain
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.RoCCIO
import framework.generic.dma.Dma

/**
 * The accelerator of a piplus thread: Frontend, MemDomain, BallDomain with the piplus Balls and the GP domain, whose RVV
 * operators run the FP32 work around the products.
 */
@instantiable
class Acc(
  accp:        AccParam,
  permissionp: PmpParams,
  simp:        SimParam)
    extends Module {
  override def desiredName = "DenseAcc"
  val frontp               = accp.frontend
  val memp                 = accp.memDomain
  val gpp                  = accp.gpDomain
  val ballp                = accp.ballDomain
  val transportp           = accp.transport

  require(gpp.enable, "piplus threads run their FP32 work on the GP domain")
  require(!memp.sharedEnable && memp.sharedPorts == 0, "piplus threads have no shared banks")

  @public
  val io = IO(new Bundle {
    val ownerId = Input(UInt(64.W))
    val debug   = new AcceleratorDebugPort(frontp, memp, gpp, ballp, simp)
    val rocc    = new RoCCIO(frontp.commandDataBits, permissionp)
    val dma     = new Dma(transportp.axi)
  })

  val frontend: Instance[Frontend] = Instantiate(new Frontend(frontp, memp, gpp, ballp, permissionp, simp))

  val ballDomain: Instance[BallDomain] =
    Instantiate(new BallDomain(
      BallLink.host(frontp, memp, gpp),
      ballp,
      permissionp,
      BallComponents.factories
    ))

  val memDomain: Instance[ThreadMemDomain] =
    Instantiate(new ThreadMemDomain(frontp, memp, gpp, transportp, permissionp, simp))

  val gpDomain: Instance[GpDomain] = Instantiate(new GpDomain(frontp, memp, gpp))

  io.debug.frontend <> frontend.io.debug
  frontend.io.ownerId          := io.ownerId
  ballDomain.io.ownerId        := io.ownerId
  memDomain.io.ownerId         := io.ownerId
  gpDomain.io.ownerId          := io.ownerId
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

  gpDomain.io.command <> frontend.io.gpRequest.get
  frontend.io.gpCompletion.get <> gpDomain.io.complete
  gpDomain.io.image <> memDomain.io.image.get
  memDomain.io.gpBanks.get <> gpDomain.io.banks
  memDomain.io.gpLayoutRequest.get <> gpDomain.io.layoutRequest
  gpDomain.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
  ballDomain.io.operator.get <> gpDomain.io.ball

  memDomain.io.banks <> ballDomain.io.banks
  memDomain.io.ballLayoutRequest <> ballDomain.io.layoutRequest
  ballDomain.io.layoutResponse <> memDomain.io.ballLayoutResponse

  frontend.io.barrier.release := false.B
  when(!reset.asBool) {
    assert(!frontend.io.barrier.arrive, "A piplus thread cannot issue a hart barrier")
  }
}
