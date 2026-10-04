package framework.balldomain.kernel

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue, KernelWriteBank}
import framework.memdomain.frontend.mem.{KernelDmaRequest, KernelMemoryBridge}
import framework.rvv.KernelEngine
import framework.memdomain.frontend.mem.dma.{DmaError, DmaStatus}

@instantiable
class KernelBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {
  require(b.rvv.enable, "KernelBall requires rvv.enable")
  private val mappings = b.ballDomain.ballIdMappings.filter(_.ballClass == "framework.balldomain.kernel.KernelBall")
  require(mappings.size == 1, "KernelBall requires exactly one generated registration")
  private val mapping  = mappings.head
  require(mapping.builtin == "kernel" && mapping.ballName == "kernel", "KernelBall requires builtin kernel metadata")
  require(
    mapping.inBW == 0 && mapping.outBW == 0 && mapping.mmioReadBW == 0 && mapping.mmioWriteBW == 0,
    "KernelBall uses only dedicated memory ports"
  )
  private val isa      = b.ballDomain.ballISA.filter(_.bid == mapping.ballId)
  require(isa.size == 1 && isa.head.funct7 == 15, "KernelBall registration must declare RUN_KERNEL funct7=15")

  @public val io = IO(new BlinkIO(b, 0, 0))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  @public val kernel_command_i  = if (b.rvv.enable) Some(IO(Flipped(Decoupled(new GlobalSchedIssue(b))))) else None
  @public val kernel_complete_o = if (b.rvv.enable) Some(IO(Decoupled(new GlobalSchedComplete(b)))) else None
  @public val kernel            = if (b.rvv.enable) Some(IO(Flipped(new KernelMemoryBridge(b)))) else None
  @public val kernelWriteBank   = IO(Valid(new KernelWriteBank(b)))
  @public val kernelFault       = IO(Output(Bool()))

  private val engine         = Instantiate(new KernelEngine(b))
  private val loadOwner      = RegInit(false.B)
  private val loadPending    = RegInit(false.B)
  private val load           = Reg(new KernelDmaRequest)
  private val loadAccepted   = RegInit(false.B)
  private val terminalSeen   = RegInit(false.B)
  private val terminalStatus = Reg(new DmaStatus)
  private val fault          = RegInit(false.B)
  private val loadCommand    = kernel_command_i.get
  private val loadComplete   = kernel_complete_o.get
  private val memory         = kernel.get

  engine.io.cmdReq.valid           := loadCommand.valid || io.cmdReq.valid
  engine.io.cmdReq.bits.cmd.bid    := mapping.ballId.U
  engine.io.cmdReq.bits.cmd.funct7 := Mux(loadCommand.valid, 12.U, io.cmdReq.bits.cmd.funct7)
  engine.io.cmdReq.bits.cmd.rs1    := Mux(loadCommand.valid, loadCommand.bits.cmd.cmd.rs1Data, io.cmdReq.bits.cmd.rs1)
  engine.io.cmdReq.bits.cmd.rs2    := Mux(loadCommand.valid, loadCommand.bits.cmd.cmd.rs2Data, io.cmdReq.bits.cmd.rs2)
  engine.io.cmdReq.bits.rob_id     := Mux(loadCommand.valid, loadCommand.bits.rob_id, io.cmdReq.bits.rob_id)
  loadCommand.ready                := engine.io.cmdReq.ready
  io.cmdReq.ready                  := engine.io.cmdReq.ready && !loadCommand.valid
  io.status                        := engine.io.status
  io.subRobReq.valid               := false.B
  io.subRobReq.bits                := SubRobRow.tieOff(b)

  when(loadCommand.fire) {
    assert(!loadCommand.bits.is_sub, "kernel load must belong to the main ROB")
    assert(loadCommand.bits.cmd.cmd.funct === 12.U, "KernelBall load bridge only accepts funct7=12")
  }
  when(io.cmdReq.fire) {
    assert(!io.cmdReq.bits.is_sub, "kernel run must belong to the main ROB")
    assert(io.cmdReq.bits.cmd.funct7 === 15.U, "KernelBall only accepts registered RUN_KERNEL")
    assert(io.cmdReq.bits.cmd.bid === mapping.ballId.U, "KernelBall run bid differs from registration")
  }
  when(engine.io.cmdReq.fire) {
    loadOwner := loadCommand.valid
    when(loadCommand.valid) {
      load.address := loadCommand.bits.cmd.cmd.rs2Data
      load.bytes   := loadCommand.bits.cmd.cmd.rs1Data(31, 0)
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
  loadComplete.valid                                                         := engine.io.cmdResp.valid && loadOwner && drained
  loadComplete.bits.rob_id                                                   := engine.io.cmdResp.bits.rob_id
  loadComplete.bits.is_sub                                                   := false.B
  loadComplete.bits.sub_rob_id                                               := 0.U
  loadComplete.bits.fault                                                    := 0.U.asTypeOf(new DmaStatus)
  when(terminalSeen && terminalStatus.error =/= DmaError.None.U && terminalStatus.error =/= DmaError.Cancelled.U) {
    loadComplete.bits.fault := terminalStatus
  }.elsewhen(engine.io.result.fault && engine.io.result.cause === 7.U) {
    loadComplete.bits.fault.error   := DmaError.Bank.U
    loadComplete.bits.fault.address := engine.io.result.tval
  }.elsewhen(engine.io.result.fault && engine.io.result.cause =/= 5.U) {
    loadComplete.bits.fault.error   := DmaError.Shape.U
    loadComplete.bits.fault.address := load.address
  }.elsewhen(terminalSeen && terminalStatus.error =/= DmaError.None.U) {
    loadComplete.bits.fault := terminalStatus
  }
  io.cmdResp.valid                                                           := engine.io.cmdResp.valid && !loadOwner
  io.cmdResp.bits.rob_id                                                     := engine.io.cmdResp.bits.rob_id
  io.cmdResp.bits.is_sub                                                     := false.B
  io.cmdResp.bits.sub_rob_id                                                 := 0.U
  engine.io.cmdResp.ready                                                    := Mux(loadOwner, loadComplete.ready, io.cmdResp.ready) && drained
  kernelWriteBank.valid                                                      := engine.io.cmdResp.fire && !loadOwner
  kernelWriteBank.bits.rob_id                                                := engine.io.cmdResp.bits.rob_id
  kernelWriteBank.bits.bank                                                  := engine.io.cmdResp.bits.write_bank
  when(engine.io.cmdResp.fire && !loadOwner && engine.io.result.fault)(fault := true.B)
  kernelFault                                                                := fault
  when(engine.io.cmdResp.valid && !loadOwner) {
    assert(
      !engine.io.result.fault,
      "RVV kernel fault: cause=%d pc=%x instruction=%x tval=%x",
      engine.io.result.cause,
      engine.io.result.pc,
      engine.io.result.instruction,
      engine.io.result.tval
    )
  }
}
