package framework.system.core.accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.rvv.{KernelEngine, RvvBallCommand}

import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.memdomain.frontend.mem.{KernelDmaRequest, KernelMemoryBridge}

import framework.memdomain.frontend.mem.dma.{DmaError, DmaStatus}

@instantiable
class Rvv(val b: GlobalConfig) extends Module {
  require(b.rvv.enable)
  @public val command      = IO(Flipped(Decoupled(new GlobalSchedIssue(b))))
  @public val complete     = IO(Decoupled(new GlobalSchedComplete(b)))
  @public val memory       = IO(Flipped(new KernelMemoryBridge(b)))
  @public val fault        = IO(Output(Bool()))
  @public val ballRequest  = IO(Decoupled(new RvvBallCommand))
  @public val ballResponse = IO(Flipped(Decoupled(UInt(64.W))))
  @public val ownerRobId   = IO(Output(UInt(log2Up(b.frontend.rob_entries).W)))
  @public val banks        = IO(Output(new framework.rvv.BankLayout(b)))
  @public val busy         = IO(Output(Bool()))

  val engine: Instance[KernelEngine] = Instantiate(new KernelEngine(b))
  val loadOwner      = RegInit(false.B)
  val loadPending    = RegInit(false.B)
  val load           = Reg(new KernelDmaRequest(b))
  val loadAccepted   = RegInit(false.B)
  val terminalSeen   = RegInit(false.B)
  val terminalStatus = Reg(new DmaStatus)
  val failed         = RegInit(false.B)
  val runOwner       = RegInit(false.B)
  val isLoad         = command.bits.cmd.cmd.funct === 44.U
  engine.io.ballRequest <> ballRequest
  engine.io.ballResponse <> ballResponse
  engine.io.cmdReq.valid             := command.valid
  engine.io.cmdReq.bits.cmd.funct7   := command.bits.cmd.cmd.funct
  engine.io.cmdReq.bits.cmd.rs1      := command.bits.cmd.cmd.rs1Data
  engine.io.cmdReq.bits.cmd.rs2      := command.bits.cmd.cmd.rs2Data
  engine.io.cmdReq.bits.rob_id       := command.bits.rob_id
  engine.io.cmdReq.bits.read_groups  := command.bits.cmd.op1_col
  engine.io.cmdReq.bits.write_groups := command.bits.cmd.wr_col
  command.ready                      := engine.io.cmdReq.ready
  busy                               := engine.io.status.running
  val owner = Reg(UInt(log2Up(b.frontend.rob_entries).W))
  ownerRobId := owner
  val layout = Reg(new framework.rvv.BankLayout(b))
  banks := layout
  when(command.fire && command.bits.cmd.cmd.funct === 79.U) {
    layout.readBank    := command.bits.cmd.cmd.rs1Data(9, 0)
    layout.writeBank   := command.bits.cmd.cmd.rs1Data(29, 20)
    layout.readGroups  := command.bits.cmd.op1_col
    layout.writeGroups := command.bits.cmd.wr_col
  }
  when(command.fire) {
    assert(!command.bits.is_sub, "RVV command must belong to the main ROB")
    owner := command.bits.rob_id
  }
  when(engine.io.cmdReq.fire) {
    loadOwner := isLoad
    runOwner  := command.bits.cmd.cmd.funct === 79.U
    when(isLoad) {
      load.address := command.bits.cmd.cmd.rs2Data
      load.bytes   := command.bits.cmd.cmd.rs1Data(63, 30)
      load.rob_id  := command.bits.rob_id
      loadPending  := true.B
      loadAccepted := false.B
      terminalSeen := false.B
    }
  }

  val cancel = loadOwner && engine.io.cmdResp.valid && engine.io.result.fault && !terminalSeen
  memory.load.valid                            := loadPending && !cancel
  memory.load.bits                             := load
  when(memory.load.fire || cancel)(loadPending := false.B)
  memory.abort                                 := cancel
  when(memory.load.fire)(loadAccepted          := true.B)
  engine.io.imageTerminal.valid                := loadOwner && loadAccepted && !terminalSeen && memory.result.valid
  engine.io.imageTerminal.bits.error           := memory.result.bits.error
  engine.io.imageTerminal.bits.address         := memory.result.bits.address
  memory.result.ready                          := loadOwner && loadAccepted && !terminalSeen && engine.io.imageTerminal.ready
  when(memory.result.fire) {
    terminalSeen   := true.B
    terminalStatus := memory.result.bits
  }
  engine.io.image <> memory.image
  memory.active                                := engine.io.status.running && !loadOwner
  engine.io.channelReady                       := memory.ready
  for (port <- 0 until b.rvv.memoryPorts) {
    memory.bankRead(port) <> engine.io.bankRead(port)
    memory.bankWrite(port) <> engine.io.bankWrite(port)
  }

  val drained = !loadOwner || (!memory.busy && !loadPending && (!loadAccepted || terminalSeen))
  complete.valid           := engine.io.cmdResp.valid && drained
  complete.bits.rob_id     := engine.io.cmdResp.bits.rob_id
  complete.bits.is_sub     := false.B
  complete.bits.sub_rob_id := 0.U
  complete.bits.fault      := 0.U.asTypeOf(new DmaStatus)
  when(
    loadOwner && terminalSeen && terminalStatus.error =/= DmaError.None.U && terminalStatus.error =/= DmaError.Cancelled.U
  ) {
    complete.bits.fault := terminalStatus
  }.elsewhen(engine.io.result.fault && (engine.io.result.cause === 7.U || !loadOwner && engine.io.result.cause === 5.U)) {
    complete.bits.fault.error   := DmaError.Bank.U
    complete.bits.fault.address := engine.io.result.tval
  }.elsewhen(engine.io.result.fault && engine.io.result.cause =/= 5.U) {
    complete.bits.fault.error   := DmaError.Shape.U
    complete.bits.fault.address := Mux(loadOwner, load.address, engine.io.result.tval)
  }.elsewhen(loadOwner && terminalSeen && terminalStatus.error =/= DmaError.None.U) {
    complete.bits.fault := terminalStatus
  }

  engine.io.cmdResp.ready                                                     := complete.ready && drained
  when(engine.io.cmdResp.fire && !loadOwner && engine.io.result.fault)(failed := true.B)
  fault                                                                       := failed
}
