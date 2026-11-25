package framework.custom.memdomain

import framework.generic.acc.memdomain._
import framework.generic.acc.configs.BallLink
import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}
import framework.generic.contract.shm.LocalBank
import framework.generic.acc.shm.BankAccess
import framework.generic.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort
import framework.generic.contract.memdomain.DmaFaultControl
import framework.generic.acc.memdomain.PhysicalBankHash
import framework.generic.arbiter.{RRArbiter}

import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.custom.debug.trace.MemoryEvent
import framework.generic.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.custom.memdomain.tomem.ToMem
import framework.custom.memdomain.toddr.ToDDR
import framework.hart.memory.{MemCtrl}
import framework.hart.memory.access.preflight.Preparation
import framework.generic.bus.axi4
import framework.custom.memdomain.bank_router.{BankMappingIO, BankRouter}
import framework.custom.memdomain.prm.PRM
import framework.custom.gpdomain.ImagePort
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.configs.BallDomainParam
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.gpdomain.configs.GpDomainParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.ball.DomainMemory
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.Response

@instantiable
class MemDomain(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val transportParam:  TransportParam,
  val permissionParam: PmpParams,
  val trackingParam:   TrackingParam,
  val walkerParam:     WalkerParam,
  val simp:            SimParam,
  val homeRanges:      Boolean)
    extends Module {
  val sharedPorts     = memParam.sharedPorts
  val diffTest        = simp.diffTest
  val commandDataBits = frontendParam.commandDataBits

  val operatorPorts =
    if (gpParam.enable)
      gpParam.memory.memoryPorts
    else 0

  val sharedHashCount = if (memParam.sharedEnable) memParam.sharedBankNum else 0

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId            = Input(UInt(64.W))
    val traceMemory        = Output(Vec(memParam.bankRequestPorts + operatorPorts + 2 + memParam.bankNum, new MemoryEvent))
// Command Channel
    val issue              = Flipped(Decoupled(new MemoryIssue(BallLink.blink(frontendParam, memParam), permissionParam)))
    val complete           = Decoupled(new Response(log2Up(frontendParam.rob_entries)))
    val mapping            = Flipped(new BankMappingIO(frontendParam))
// Inside Channel
    val banks              = Vec(memParam.bankRequestPorts, Flipped(new BankPort(BallLink.blink(frontendParam, memParam))))
    val ballLayoutRequest  = Flipped(Decoupled(new BankLayoutRequest(frontendParam.bank_id_len)))
    val ballLayoutResponse = Decoupled(new BankLayout(memParam.groupCountWidth))
    val gpLayoutRequest    =
      Option.when(gpParam.enable)(Flipped(Decoupled(new BankLayoutRequest(frontendParam.bank_id_len))))
    val gpLayoutResponse   = Option.when(gpParam.enable)(Decoupled(new BankLayout(memParam.groupCountWidth)))
    val gpBanks            =
      Option.when(gpParam.enable)(Vec(operatorPorts, Flipped(new BankPort(BallLink.blink(frontendParam, memParam)))))
    val image              = Option.when(gpParam.enable)(new ImagePort)
// Outside Channel
    val ball               = Flipped(new DomainMemory(BallLink.blink(frontendParam, memParam), permissionParam))
    val memory             = new MemCtrl(trackingParam, walkerParam)
    val fault              = new DmaFaultControl
    val ddr                = new axi4.Port(transportParam.axi)
    val readAddress        = Output(UInt(64.W))
    val writeAddress       = Output(UInt(64.W))

    val shared = new Bundle {
      val bank = new BankAccess(frontendParam, memParam)
      val move = new MvoverPort
    }

    val bank_hashes =
      if (diffTest) {
        Some(Output(Vec(memParam.bankNum + sharedHashCount, new PhysicalBankHash(frontendParam))))
      } else {
        None
      }

    val shared_bank_hashes =
      if (diffTest && memParam.sharedEnable) {
        Some(Input(Vec(sharedHashCount, new PhysicalBankHash(frontendParam))))
      } else {
        None
      }

  })

  val control: Instance[MemoryControl] =
    Instantiate(new MemoryControl(gpParam, transportParam, permissionParam, trackingParam, walkerParam, homeRanges))

  val preparation: Instance[Preparation] = Instantiate(new Preparation(
    framework.hart.memory.access.preflight.configs.Config(
      permissionParam,
      transportParam.mapping,
      transportParam.regions
    ),
    walkerParam
  ))

  val transport = Instantiate(new ToDDR(transportParam, permissionParam))

  val ctrl:       Instance[Ctrl]       = Instantiate(new Ctrl(frontendParam, memParam, permissionParam))
  val toMem:      Instance[ToMem]      =
    Instantiate(new ToMem(frontendParam, memParam, gpParam, transportParam, permissionParam))
  val bankRouter: Instance[BankRouter] =
    Instantiate(new BankRouter(frontendParam, memParam, gpParam, simp.diffTest))
  val prm:        Instance[PRM]        = Instantiate(new PRM(memParam, simp.diffTest))
  ctrl.io.ownerId       := io.ownerId
  toMem.io.ownerId      := io.ownerId
  bankRouter.io.ownerId := io.ownerId

  io.traceMemory                 := bankRouter.io.traceMemory
  prm.io.banks <> bankRouter.io.physical
  bankRouter.io.mapping <> io.mapping
  io.shared.bank.mapping <> bankRouter.io.shared.mapping
  bankRouter.io.hashes.foreach(_ := prm.io.hashes.get)
  io.bank_hashes.foreach { hashes =>
    for (i <- 0 until memParam.bankNum)
      hashes(i) := bankRouter.io.bankHashes.get(i)
    for (i <- 0 until sharedHashCount)
      hashes(memParam.bankNum + i) := io.shared_bank_hashes.get(i)
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
    new MemoryIssue(BallLink.blink(frontendParam, memParam), permissionParam),
    2
  ))

  commands.io.in(0) <> io.issue
  commands.io.in(1) <> io.ball.request
  ctrl.io.issue <> commands.io.out
  val childOwner = RegInit(VecInit(Seq.fill(frontendParam.rob_entries)(false.B)))
  when(ctrl.io.issue.fire)(childOwner(ctrl.io.issue.bits.operation.rob_id) := commands.io.chosen === 1.U)
  val child = childOwner(ctrl.io.complete.bits.rob_id)
  io.complete.valid      := ctrl.io.complete.valid && !child
  io.complete.bits       := ctrl.io.complete.bits
  io.ball.response.valid := ctrl.io.complete.valid && child
  io.ball.response.bits  := ctrl.io.complete.bits
  ctrl.io.complete.ready := Mux(child, io.ball.response.ready, io.complete.ready)
  io.ball.metadata <> toMem.io.metadata
  io.shared.move <> ctrl.io.mvover
  ctrl.io.request <> toMem.io.request
  toMem.io.response <> ctrl.io.response

  control.io.memory <> toMem.io.memory.control
  control.io.preparation <> preparation.io
  io.memory <> control.io.service
  io.fault <> control.io.fault
  transport.io.dma <> toMem.io.dma
  transport.io.decisions := control.io.decisions
  control.io.queries     := transport.io.queries
  transport.io.mappings  := control.io.mappings
  io.ddr <> transport.io.axi
  io.readAddress         := transport.io.readAddress
  io.writeAddress        := transport.io.writeAddress

  bankRouter.io.requests <> io.banks
  bankRouter.io.ballLayoutRequest <> io.ballLayoutRequest
  io.ballLayoutResponse <> bankRouter.io.ballLayoutResponse
  io.gpLayoutRequest.zip(bankRouter.io.gpLayoutRequest).foreach { case (source, sink) => sink <> source }
  io.gpLayoutResponse.zip(bankRouter.io.gpLayoutResponse).foreach { case (sink, source) => sink <> source }

  io.image.zip(toMem.io.image).foreach { case (sink, source) => sink <> source }
  io.gpBanks.foreach { ports =>
    for (port <- 0 until operatorPorts) bankRouter.io.directRequest(port) <> ports(port)
  }

  bankRouter.io.directRequest(operatorPorts) <> toMem.io.bankRead
  bankRouter.io.directRequest(operatorPorts + 1) <> toMem.io.bankWrite
  bankRouter.io.shared.local <> io.shared.bank.local
  bankRouter.io.config <> ctrl.io.config

  // Shared path passthrough
  io.shared.bank.requests <> bankRouter.io.shared.requests
  io.shared.bank.config <> bankRouter.io.shared.config
}
