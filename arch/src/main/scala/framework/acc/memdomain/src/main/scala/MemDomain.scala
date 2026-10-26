package framework.acc.memdomain

import sims.configs.SimParam
import framework.lib.arbiter.{RRArbiter}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.acc.debug.trace.MemoryEvent
import framework.ip.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.scheduler.MemoryIssue
import framework.acc.balldomain.DomainMemory
import framework.acc.memdomain.tomem.ToMem
import framework.core.memory.MemoryPort
import framework.core.memory.access.preflight.Preparation
import framework.ip.bus.axi4
import framework.acc.memdomain.bank_router.{BankMapping, BankRouter}
import framework.acc.memdomain.prm.PRM
import framework.acc.memdomain.{MvoverISA, MvoverPort}
import framework.ip.memory.mesh_shm.MeshLocalBankPort
import framework.acc.gpdomain.ImagePort
import framework.thread.tlink.BankAccess

class PhysicalBankHash(val frontendParam: FrontendParam) extends Bundle {
  val valid      = Bool()
  val threadId   = UInt(64.W)
  val vbankId    = UInt(frontendParam.bank_id_len.W)
  val pbankId    = UInt(32.W)
  val groupId    = UInt(32.W)
  val statusHash = UInt(32.W)
}

class BankRead(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Bundle {
  val bank_id  = Input(UInt(frontendParam.bank_id_len.W))
  val rob_id   = Input(UInt(log2Up(frontendParam.rob_entries).W))
  val group_id = Input(UInt(memParam.groupIdWidth.W))

  val io = new SramReadIO(memParam)
}

class BankWrite(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Bundle {
  val bank_id  = Input(UInt(frontendParam.bank_id_len.W))
  val rob_id   = Input(UInt(log2Up(frontendParam.rob_entries).W))
  val group_id = Input(UInt(memParam.groupIdWidth.W))

  val io = new SramWriteIO(memParam)
}

class BankPort(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val write     = Flipped(new SramWriteIO(memParam))
  val read      = Flipped(new SramReadIO(memParam))
  val bank_id   = Output(UInt(frontendParam.bank_id_len.W))
  val group_id  = Output(UInt(memParam.groupIdWidth.W))
  val is_shared = Output(Bool())
  val thread_id = Output(UInt(64.W))
  val rob_id    = Output(UInt(log2Up(frontendParam.rob_entries).W))
  val inst_id   = Output(UInt(64.W))
}

class LayoutQuery(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val bank   = Output(UInt(frontendParam.bank_id_len.W))
  val groups = Input(UInt(memParam.groupCountWidth.W))
}

class BankLayoutRequest(frontendParam: FrontendParam) extends Bundle {
  val banks   = Vec(3, UInt(frontendParam.bank_id_len.W))
  val enabled = Vec(3, Bool())
}

class BankLayout(memParam: MemDomainParam) extends Bundle {
  val groups = Vec(3, UInt(memParam.groupCountWidth.W))
}

@instantiable
class MemDomain(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val transportParam:  TransportParam,
  val permissionParam: PmpParams,
  val trackingParam:   TrackingParam,
  val walkerParam:     WalkerParam,
  val simp:            SimParam)
    extends Module {
  val sharedPorts     = memParam.sharedPorts
  val diffTest        = simp.diffTest
  val commandDataBits = frontendParam.commandDataBits
  val ownerId         = frontendParam.ownerId

  val kernelPorts =
    if (gpParam.enable)
      gpParam.memory.memoryPorts
    else 0

  val sharedHashCount = if (memParam.sharedEnable) memParam.sharedBankNum else 0

  @public
  val io = IO(new Bundle {
    val traceMemory        = Output(Vec(memParam.bankRequestPorts + kernelPorts + 2 + memParam.bankNum, new MemoryEvent))
// Command Channel
    val issue              = Flipped(Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam)))
    val complete           = Decoupled(new Response(frontendParam))
    val mapping            = Flipped(Decoupled(new BankMapping(frontendParam)))
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
    val memory             = new MemoryPort(trackingParam, walkerParam)
    val ddr                = new axi4.Port(transportParam.axi)

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
    Instantiate(new MemoryControl(gpParam, transportParam, permissionParam, trackingParam, walkerParam))

  val preparation: Instance[Preparation] = Instantiate(new Preparation(
    framework.core.memory.access.preflight.configs.Config(
      permissionParam,
      transportParam.mapping,
      transportParam.regions
    ),
    walkerParam
  ))

  val ctrl:       Instance[Ctrl]       = Instantiate(new Ctrl(frontendParam, memParam, permissionParam))
  val toMem:      Instance[ToMem]      =
    Instantiate(new ToMem(frontendParam, memParam, gpParam, transportParam, permissionParam))
  val bankRouter: Instance[BankRouter] =
    Instantiate(new BankRouter(frontendParam, memParam, gpParam, simp.diffTest))
  val prm:        Instance[PRM]        = Instantiate(new PRM(memParam, simp.diffTest))

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
    new MemoryIssue(frontendParam, memParam, permissionParam),
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
  io.ddr <> toMem.io.memory.axi

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
  bankRouter.io.shared.local <> io.shared.bank.local
  bankRouter.io.config <> ctrl.io.config

  // Shared path passthrough
  io.shared.bank.requests <> bankRouter.io.shared.requests
  io.shared.bank.config <> bankRouter.io.shared.config
}
