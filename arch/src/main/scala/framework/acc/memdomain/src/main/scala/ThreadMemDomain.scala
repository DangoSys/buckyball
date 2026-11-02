package framework.acc.memdomain

import sims.configs.SimParam
import framework.lib.arbiter.{RRArbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.acc.debug.trace.MemoryEvent

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.scheduler.MemoryIssue
import framework.acc.balldomain.DomainMemory
import framework.acc.memdomain.tomem.ToMem
import framework.acc.memdomain.bank_router.{BankMappingIO, BankRouter}
import framework.acc.memdomain.prm.PRM
import framework.acc.gpdomain.ImagePort

@instantiable
class ThreadMemDomain(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val transportParam:  TransportParam,
  val permissionParam: PmpParams,
  val simp:            SimParam)
    extends Module {
  val diffTest = simp.diffTest

  val kernelPorts =
    if (gpParam.enable)
      gpParam.memory.memoryPorts
    else 0

  require(!memParam.sharedEnable && memParam.sharedPorts == 0)

  @public
  val io = IO(new Bundle {
    val traceMemory        = Output(Vec(memParam.bankRequestPorts + kernelPorts + 2 + memParam.bankNum, new MemoryEvent))
// Command Channel
    val issue              = Flipped(Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam)))
    val complete           = Decoupled(new Response(frontendParam))
    val mapping            = Flipped(new BankMappingIO(frontendParam))
// Inside Channel
    val banks              = Vec(memParam.bankRequestPorts, Flipped(new BankPort(frontendParam, memParam)))
    val ballLayoutRequest  = Flipped(Decoupled(new BankLayoutRequest(frontendParam)))
    val ballLayoutResponse = Decoupled(new BankLayout(memParam))
    val gpLayoutRequest    = Option.when(gpParam.enable)(Flipped(Decoupled(new BankLayoutRequest(frontendParam))))
    val gpLayoutResponse   = Option.when(gpParam.enable)(Decoupled(new BankLayout(memParam)))
    val gpBanks            = Option.when(gpParam.enable)(Vec(kernelPorts, Flipped(new BankPort(frontendParam, memParam))))
    val image              = Option.when(gpParam.enable)(new ImagePort)
// Outside Channel
    val ball               = Flipped(new DomainMemory(frontendParam, memParam, permissionParam))
    val dma                = new framework.thread.tlink.Dma(transportParam.axi)

    val bank_hashes =
      if (diffTest) {
        Some(Output(Vec(memParam.bankNum, new PhysicalBankHash(frontendParam))))
      } else {
        None
      }

  })

  val transport = Instantiate(new framework.thread.tlink.Transport(transportParam, permissionParam))

  val ctrl:       Instance[Ctrl]       = Instantiate(new Ctrl(frontendParam, memParam, permissionParam))
  val toMem:      Instance[ToMem]      =
    Instantiate(new ToMem(frontendParam, memParam, gpParam, transportParam, permissionParam))
  val bankRouter: Instance[BankRouter] =
    Instantiate(new BankRouter(frontendParam, memParam, gpParam, simp.diffTest))
  val prm:        Instance[PRM]        = Instantiate(new PRM(memParam, simp.diffTest))

  io.traceMemory                     := bankRouter.io.traceMemory
  prm.io.banks <> bankRouter.io.physical
  bankRouter.io.mapping <> io.mapping
  bankRouter.io.shared.mapping.ready := true.B
  bankRouter.io.hashes.foreach(_     := prm.io.hashes.get)
  io.bank_hashes.foreach { hashes =>
    for (i <- 0 until memParam.bankNum)
      hashes(i) := bankRouter.io.bankHashes.get(i)
  }

  // Query the router layout.
  bankRouter.io.queryVbank  := toMem.io.queryVbank
  bankRouter.io.queryShared := toMem.io.queryShared
  toMem.io.queryGroups      := bankRouter.io.queryGroups
  ctrl.io.clearBusy         := bankRouter.io.clearBusy

//===----------------------------------------------------------------------===//
// Control and transport connections
//===----------------------------------------------------------------------===//
  val commands = Instantiate(new RRArbiter(
    new MemoryIssue(frontendParam, memParam, permissionParam),
    2
  ))

  commands.io.in(0) <> io.issue
  commands.io.in(1) <> io.ball.request
  ctrl.io.issue <> commands.io.out
  val childOwner = RegInit(VecInit(Seq.fill(frontendParam.rob_entries)(false.B)))
  when(ctrl.io.issue.fire)(childOwner(ctrl.io.issue.bits.operation.rob_id) := commands.io.chosen === 1.U)
  val child = childOwner(ctrl.io.complete.bits.rob_id)
  io.complete.valid               := ctrl.io.complete.valid && !child
  io.complete.bits                := ctrl.io.complete.bits
  io.ball.response.valid          := ctrl.io.complete.valid && child
  io.ball.response.bits           := ctrl.io.complete.bits
  ctrl.io.complete.ready          := Mux(child, io.ball.response.ready, io.complete.ready)
  io.ball.metadata <> toMem.io.metadata
  ctrl.io.mvover.command.ready    := false.B
  ctrl.io.mvover.completion.valid := false.B
  ctrl.io.mvover.completion.bits  := false.B
  when(!reset.asBool)(assert(!ctrl.io.mvover.command.valid, "Thread cannot issue a core bank move"))
  ctrl.io.request <> toMem.io.request
  toMem.io.response <> ctrl.io.response

  transport.io.dma <> toMem.io.dma
  io.dma <> transport.io.link
  for (i <- 0 until 3) {
    toMem.io.memory.control.complete(i).valid := toMem.io.memory.control.finished(i).valid
    toMem.io.memory.control.complete(i).bits  := toMem.io.memory.control.finished(i).bits
  }

  bankRouter.io.requests <> io.banks
  bankRouter.io.ballLayoutRequest <> io.ballLayoutRequest
  io.ballLayoutResponse <> bankRouter.io.ballLayoutResponse
  io.gpLayoutRequest.zip(bankRouter.io.gpLayoutRequest).foreach { case (source, sink) => sink <> source }
  io.gpLayoutResponse.zip(bankRouter.io.gpLayoutResponse).foreach { case (sink, source) => sink <> source }

  io.image.zip(toMem.io.image).foreach { case (sink, source) => sink <> source }
  io.gpBanks.foreach { ports =>
    for (port <- 0 until kernelPorts) bankRouter.io.directRequest(port) <> ports(port)
  }

  bankRouter.io.directRequest(kernelPorts) <> toMem.io.bankRead
  bankRouter.io.directRequest(kernelPorts + 1) <> toMem.io.bankWrite
  bankRouter.io.shared.local.request.valid  := false.B
  bankRouter.io.shared.local.request.bits   := 0.U.asTypeOf(bankRouter.io.shared.local.request.bits)
  bankRouter.io.shared.local.response.ready := true.B
  bankRouter.io.config <> ctrl.io.config

  // Shared path passthrough
  bankRouter.io.shared.config.ready := true.B
}
