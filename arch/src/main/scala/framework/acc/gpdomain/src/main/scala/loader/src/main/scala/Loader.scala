package framework.acc.gpdomain.loader

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.acc.frontend.scheduler.{Complete, Issue}
import framework.acc.memdomain.toddr.DmaStatus
import framework.acc.gpdomain.execution.{
  KernelCompletion,
  KernelLaunch,
  ProgramWrite,
  VectorMemoryRequest,
  VectorMemoryResponse
}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

class KernelContext(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val instId      = UInt(64.W)
  val rob_id      = UInt(log2Ceil(frontendParam.rob_entries).W)
  val readBank    = UInt(frontendParam.bank_id_len.W)
  val writeBank   = UInt(frontendParam.bank_id_len.W)
  val readGroups  = UInt(memParam.groupCountWidth.W)
  val writeGroups = UInt(memParam.groupCountWidth.W)
}

class LoaderMemoryPort(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val request    = Decoupled(new VectorMemoryRequest)
  val response   = Flipped(Decoupled(new VectorMemoryResponse))
  val control    = Output(Bool())
  val buffer     = Output(Bool())
  val loading    = Output(Bool())
  val constBytes = Output(UInt(32.W))
  val context    = Output(new KernelContext(frontendParam, memParam))
}

@instantiable
class Loader(val frontendParam: FrontendParam, val memParam: MemDomainParam, val gpParam: GpDomainParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  require(gpParam.enable, "GP domain must be explicitly enabled")

  @public
  val io = IO(new Bundle {
    val running        = Output(Bool())
    val cmdReq         = Flipped(Decoupled(new Issue(frontendParam, memParam)))
    val cmdResp        = Decoupled(new Complete(frontendParam))
    val uploadStart    = Flipped(Decoupled(new ImageLoad))
    val uploadComplete = Decoupled(new KernelCompletion)
    val release        = Flipped(Valid(Bool()))
    val image          = Flipped(Decoupled(UInt(32.W)))
    val imageTerminal  = Flipped(Decoupled(new DmaStatus))
    val program        = Decoupled(new ProgramWrite)
    val launch         = Decoupled(new KernelLaunch(frontendParam, memParam))
    val done           = Flipped(Decoupled(new KernelCompletion))
    val memory         = new LoaderMemoryPort(frontendParam, memParam)
  })

  val request   = io.cmdReq
  val uploading = RegInit(false.B)

  val parser: Instance[ImageLoader] = Instantiate(new ImageLoader(gpParam))
  val idle :: loading :: descriptorSend :: descriptorWait :: launch :: executing :: completed :: Nil = Enum(7)

  val state             = RegInit(idle)
  val command           = Reg(new Issue(frontendParam, memParam))
  val validImage        = RegInit(VecInit(Seq.fill(2)(false.B)))
  val imageEntry        = Reg(Vec(2, UInt(32.W)))
  val imageBytes        = Reg(Vec(2, UInt(32.W)))
  val imageConstants    = Reg(Vec(2, UInt(32.W)))
  val buffer            = Reg(Bool())
  val descriptor        = Reg(Vec(12, UInt(64.W)))
  val word              = Reg(UInt(4.W))
  val terminalSeen      = RegInit(false.B)
  val terminalFailed    = RegInit(false.B)
  val terminalCancelled = RegInit(false.B)
  val result            = RegInit(0.U.asTypeOf(new KernelCompletion))

  request.ready        := state === idle && !io.uploadStart.valid
  io.uploadStart.ready := state === idle
  val completedImage = state === completed && (!uploading || result.fault || terminalSeen)
  io.cmdResp.valid              := completedImage && !uploading
  io.uploadComplete.valid       := completedImage && uploading
  io.uploadComplete.bits        := result
  io.cmdResp.bits.rob_id        := command.rob_id
  when(io.cmdResp.valid)(assert(!result.fault, "GP task violated its execution contract"))
  io.running                    := state =/= idle
  io.memory.context.instId      := command.instId
  io.memory.context.rob_id      := command.rob_id
  io.memory.context.readBank    := command.cmd.cmd.rs1Data(9, 0)
  io.memory.context.writeBank   := command.cmd.cmd.rs1Data(29, 20)
  io.memory.context.readGroups  := command.cmd.op1_col
  io.memory.context.writeGroups := command.cmd.wr_col
  io.memory.control             := state === loading || state === descriptorSend || state === descriptorWait
  io.memory.buffer              := buffer
  io.memory.loading             := state === loading
  io.memory.constBytes          := imageConstants(buffer)

  parser.io.start.valid  := io.uploadStart.fire
  parser.io.start.bits   := io.uploadStart.bits
  parser.io.image <> io.image
  io.imageTerminal.ready := state =/= idle && uploading && !terminalSeen
  val terminalError = io.imageTerminal.fire && io.imageTerminal.bits.error =/= 0.U
  parser.io.abort        := terminalError
  when(io.imageTerminal.fire) {
    terminalSeen      := true.B
    terminalFailed    := io.imageTerminal.bits.error =/= 0.U
    terminalCancelled := io.imageTerminal.bits.error === 10.U
    when(terminalError) {
      validImage(buffer) := false.B
      when(!result.fault) {
        result.fault := true.B
        result.cause := 5.U
        result.tval  := io.imageTerminal.bits.address
      }
    }
  }
  parser.io.done.ready   := state === loading
  io.program <> parser.io.program
  io.launch.valid        := state === launch
  io.launch.bits.context := io.memory.context
  io.launch.bits.iBuffer := buffer
  io.launch.bits.entry   := descriptor(0)(31, 0)
  io.launch.bits.end     := descriptor(1)(31, 0)
  io.launch.bits.stack   := descriptor(2)
  for (i <- 0 until 8) (io.launch.bits.args(i) := descriptor(i + 3))
  io.done.ready := state === executing

  io.memory.request.valid                                 := state === loading && parser.io.memoryRequest.valid
  io.memory.request.bits                                  := parser.io.memoryRequest.bits
  parser.io.memoryRequest.ready                           := state === loading && io.memory.request.ready
  parser.io.memoryResponse.valid                          := state === loading && io.memory.response.valid
  parser.io.memoryResponse.bits                           := io.memory.response.bits
  io.memory.response.ready                                := state === loading && parser.io.memoryResponse.ready
  when(state === descriptorSend) {
    io.memory.request.valid        := true.B
    io.memory.request.bits.address := (command.cmd.op1_col.pad(64) << 32) + command.cmd.cmd.rs2Data + word * 8.U
    io.memory.request.bits.write   := false.B
    io.memory.request.bits.data    := 0.U
    io.memory.request.bits.mask    := 255.U
    io.memory.request.bits.size    := 3.U
  }
  when(state === descriptorWait)(io.memory.response.ready := true.B)

  when(io.release.valid)(validImage(io.release.bits)             := false.B)
  when(io.uploadStart.fire) {
    uploading                              := true.B
    command                                := 0.U.asTypeOf(command)
    buffer                                 := io.uploadStart.bits.buffer
    validImage(io.uploadStart.bits.buffer) := false.B
    result                                 := 0.U.asTypeOf(result)
    terminalSeen                           := false.B
    terminalFailed                         := false.B
    terminalCancelled                      := false.B
    state                                  := loading
  }
  when(request.fire) {
    assert(request.bits.cmd.cmd.raw_inst(31, 25) === 79.U, "Loader launch requires RUN_KERNEL")
    uploading         := false.B
    command           := request.bits
    result            := 0.U.asTypeOf(result)
    terminalSeen      := false.B
    terminalFailed    := false.B
    terminalCancelled := false.B
    val program      = request.bits.cmd.cmd.rs1Data(19, 10)
    val programValid =
      program >= memParam.virtualBankCount.U && program < (memParam.virtualBankCount + 2).U
    val slot         = program - memParam.virtualBankCount.U
    buffer := slot(0)
    word   := 0.U
    state  := descriptorSend
    when(
      !programValid || !validImage(slot(0)) || request.bits.cmd.op1_col === 0.U ||
        request.bits.cmd.wr_col === 0.U || request.bits.cmd.cmd.rs1Data(63, 30).orR ||
        request.bits.cmd.cmd.rs1Data(9, 0) === request.bits.cmd.cmd.rs1Data(29, 20) ||
        request.bits.cmd.cmd.rs2Data(63, 32).orR ||
        (request.bits.cmd.cmd.rs2Data(
          31,
          0
        ) +& 96.U) > (memParam.bankEntries * (memParam.bankWidth / 8)).U
    ) {
      result.fault := true.B
      result.cause := 1.U
      result.tval  := request.bits.cmd.cmd.rs1Data
      state        := completed
    }
  }
  when(parser.io.done.fire) {
    val nativeFault = parser.io.done.bits.fault && parser.io.done.bits.cause =/= 5.U
    when((!terminalFailed && !terminalError) ||
      (nativeFault && (terminalCancelled || (io.imageTerminal.fire && io.imageTerminal.bits.error === 10.U)))) {
      result.fault := parser.io.done.bits.fault
      result.cause := parser.io.done.bits.cause
      result.tval  := parser.io.done.bits.tval
    }
    imageEntry(buffer) := parser.io.done.bits.entry
    imageBytes(buffer)     := parser.io.done.bits.textBytes
    imageConstants(buffer) := parser.io.done.bits.constBytes
    state                  := completed
  }
  when(state === descriptorSend && io.memory.request.fire)(state := descriptorWait)
  when(state === descriptorWait && io.memory.response.fire) {
    when(io.memory.response.bits.error) {
      result.fault := true.B
      result.cause := 5.U
      result.tval  := (command.cmd.op1_col.pad(64) << 32) + command.cmd.cmd.rs2Data + word * 8.U
      state        := completed
    }.otherwise {
      descriptor(word) := io.memory.response.bits.data
      word             := word + 1.U
      state            := Mux(word === 11.U, launch, descriptorSend)
    }
  }
  when(state === launch) {
    when(descriptor(0) =/= imageEntry(buffer) || descriptor(1) =/= imageBytes(buffer) || descriptor(
      11
    ) =/= 0.U || descriptor(2) =/= "h40002000".U) {
      io.launch.valid := false.B
      result.fault    := true.B
      result.cause    := 1.U
      result.tval     := descriptor(0)
      state           := completed
    }.elsewhen(io.launch.fire)(state := executing)
  }
  when(io.done.fire) {
    result := io.done.bits
    state  := completed
  }
  when(io.cmdResp.fire || io.uploadComplete.fire) {
    when(uploading && !result.fault && terminalSeen && !terminalFailed) {
      validImage(buffer) := true.B
    }
    state := idle
  }
}
