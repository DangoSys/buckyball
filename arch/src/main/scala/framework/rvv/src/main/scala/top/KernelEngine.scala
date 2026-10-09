package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, HasBallStatus}
import framework.top.GlobalConfig

@instantiable
class KernelEngine(val b: GlobalConfig) extends Module with HasBallStatus {
  val p = b.rvv
  require(p.enable, "RVV must be explicitly enabled")

  @public
  val io = IO(new KernelBlinkIO(b))
  def status: BallStatus = io.status

  val execution: Instance[Execution]     = Instantiate(new Execution(b))
  val loader:    Instance[ImageLoader]   = Instantiate(new ImageLoader(b))
  val local:     Instance[KernelMemory]  = Instantiate(new KernelMemory(b))
  val ports:     Seq[Instance[BankPort]] = Seq.fill(p.memoryPorts)(Instantiate(new BankPort(b)))
  val idle :: loading :: descriptorSend :: descriptorWait :: launch :: executing :: completed :: Nil = Enum(7)

  val state             = RegInit(idle)
  val command           = Reg(new KernelRequest(b))
  val channels          = RegInit(false.B)
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

  io.cmdReq.ready                                  := state === idle
  io.cmdResp.valid                                 := state === completed && (command.cmd.funct7 =/= 44.U || result.fault || terminalSeen)
  io.cmdResp.bits.rob_id                           := command.rob_id
  io.cmdResp.bits.write_bank                       := Mux(command.cmd.funct7 === 79.U && !result.fault, command.cmd.rs1(29, 20), 0.U)
  io.result                                        := result
  io.status.idle                                   := state === idle
  io.status.running                                := state =/= idle
  when(state =/= idle && io.channelReady)(channels := true.B)

  val loadBank      = io.cmdReq.bits.cmd.rs1(29, 20)
  val loadBankValid = loadBank >= b.memDomain.virtualBankCount.U && loadBank < (b.memDomain.virtualBankCount + 2).U
  val loadBuffer    = loadBank - b.memDomain.virtualBankCount.U
  loader.io.start.valid       := io.cmdReq.fire && io.cmdReq.bits.cmd.funct7 === 44.U && loadBankValid
  loader.io.start.bits.buffer := loadBuffer(0)
  loader.io.start.bits.bytes  := io.cmdReq.bits.cmd.rs1(63, 30)
  loader.io.image <> io.image
  io.imageTerminal.ready      := state =/= idle && command.cmd.funct7 === 44.U && !terminalSeen
  val terminalError = io.imageTerminal.fire && io.imageTerminal.bits.error =/= 0.U
  loader.io.abort                  := terminalError
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
  loader.io.done.ready             := state === loading
  io.ballRequest <> execution.io.ballRequest
  execution.io.ballResponse <> io.ballResponse
  execution.io.program <> loader.io.program
  execution.io.launch.valid        := state === launch
  execution.io.launch.bits.iBuffer := buffer
  execution.io.launch.bits.entry   := descriptor(0)(31, 0)
  execution.io.launch.bits.end     := descriptor(1)(31, 0)
  execution.io.launch.bits.stack   := descriptor(2)
  for (i <- 0 until 8) (execution.io.launch.bits.args(i) := descriptor(i + 3))
  execution.io.done.ready := state === executing

  local.io.buffer     := buffer
  local.io.loading    := state === loading
  local.io.constBytes := imageConstants(buffer)
  for (i <- 0 until p.memoryPorts) {
    io.bankRead(i) <> ports(i).io.read
    io.bankWrite(i) <> ports(i).io.write
    ports(i).io.robId       := command.rob_id
    ports(i).io.readBank    := command.cmd.rs1(9, 0)
    ports(i).io.writeBank   := command.cmd.rs1(29, 20)
    ports(i).io.readGroups  := command.read_groups
    ports(i).io.writeGroups := command.write_groups
    val address  = execution.io.memoryRequest(i).bits.address
    val internal = address >= "h40000000".U && address < "h40002000".U
    ports(i).io.request.valid            := state === executing && channels && !internal && execution.io.memoryRequest(i).valid
    ports(i).io.request.bits             := execution.io.memoryRequest(i).bits
    local.io.request(i).valid            := state === executing && internal && execution.io.memoryRequest(i).valid
    local.io.request(i).bits             := execution.io.memoryRequest(i).bits
    execution.io.memoryRequest(i).ready  := state === executing && Mux(
      internal,
      local.io.request(i).ready,
      channels && ports(i).io.request.ready
    )
    execution.io.memoryResponse(i).valid := state === executing && (local.io.response(i).valid || ports(
      i
    ).io.response.valid)
    execution.io.memoryResponse(i).bits  := Mux(
      local.io.response(i).valid,
      local.io.response(i).bits,
      ports(i).io.response.bits
    )
    ports(i).io.response.ready           := state === executing && execution.io.memoryResponse(i).ready
    local.io.response(i).ready           := state === executing && execution.io.memoryResponse(i).ready
  }
  loader.io.memoryRequest.ready := state === loading && local.io.request(0).ready
  loader.io.memoryResponse.valid                            := state === loading && local.io.response(0).valid
  loader.io.memoryResponse.bits                             := local.io.response(0).bits
  when(state === loading) {
    local.io.request(0).valid  := loader.io.memoryRequest.valid
    local.io.request(0).bits   := loader.io.memoryRequest.bits
    local.io.response(0).ready := loader.io.memoryResponse.ready
  }
  when(state === descriptorSend) {
    ports(0).io.request.valid        := channels
    ports(0).io.request.bits.address := (command.read_groups.pad(64) << 32) + command.cmd.rs2 + word * 8.U
    ports(0).io.request.bits.write   := false.B
    ports(0).io.request.bits.data    := 0.U
    ports(0).io.request.bits.mask    := 255.U
    ports(0).io.request.bits.size    := 3.U
  }
  when(state === descriptorWait)(ports(0).io.response.ready := true.B)

  when(io.cmdReq.fire) {
    command           := io.cmdReq.bits
    channels          := io.channelReady
    result            := 0.U.asTypeOf(new KernelCompletion)
    terminalSeen      := false.B
    terminalFailed    := false.B
    terminalCancelled := false.B
    switch(io.cmdReq.bits.cmd.funct7) {
      is(44.U) {
        buffer                    := loadBuffer(0)
        validImage(loadBuffer(0)) := false.B
        state                     := loading
        when(!loadBankValid || io.cmdReq.bits.cmd.rs1(19, 0).orR) {
          result.fault := true.B
          result.cause := 2.U
          result.tval  := io.cmdReq.bits.cmd.rs1
          state        := completed
        }
      }
      is(79.U) {
        val program      = io.cmdReq.bits.cmd.rs1(19, 10)
        val programValid = program >= b.memDomain.virtualBankCount.U && program < (b.memDomain.virtualBankCount + 2).U
        val slot         = program - b.memDomain.virtualBankCount.U
        buffer := slot(0)
        word   := 0.U
        state  := descriptorSend
        when(!programValid || !validImage(
          slot(0)
        ) || io.cmdReq.bits.read_groups === 0.U || io.cmdReq.bits.write_groups === 0.U ||
          io.cmdReq.bits.cmd.rs1(63, 30).orR || io.cmdReq.bits.cmd.rs1(9, 0) === io.cmdReq.bits.cmd.rs1(29, 20) ||
          io.cmdReq.bits.cmd.rs2(63, 32).orR || (io.cmdReq.bits.cmd.rs2(
            31,
            0
          ) +& 96.U) > (b.memDomain.bankEntries * (b.memDomain.bankWidth / 8)).U) {
          result.fault := true.B
          result.cause := 1.U
          result.tval  := io.cmdReq.bits.cmd.rs1
          state        := completed
        }
      }
      is(32.U) {
        val program = io.cmdReq.bits.cmd.rs1(9, 0)
        val slot    = program - b.memDomain.virtualBankCount.U
        when(program < b.memDomain.virtualBankCount.U || program >= (b.memDomain.virtualBankCount + 2).U ||
          io.cmdReq.bits.cmd.rs1(63, 10).orR || io.cmdReq.bits.cmd.rs2.orR) {
          result.fault := true.B
          result.cause := 2.U
          result.tval  := io.cmdReq.bits.cmd.rs1
        }.otherwise(validImage(slot(0)) := false.B)
        state := completed
      }

    }
    when(io.cmdReq.bits.cmd.funct7 =/= 44.U && io.cmdReq.bits.cmd.funct7 =/= 79.U && io.cmdReq.bits.cmd.funct7 =/= 32.U) {
      result.fault := true.B
      result.cause := 2.U
      result.tval  := io.cmdReq.bits.cmd.funct7
      state        := completed
    }
  }
  when(loader.io.done.fire) {
    val nativeFault = loader.io.done.bits.fault && loader.io.done.bits.cause =/= 5.U
    when((!terminalFailed && !terminalError) ||
      (nativeFault && (terminalCancelled || (io.imageTerminal.fire && io.imageTerminal.bits.error === 10.U)))) {
      result.fault := loader.io.done.bits.fault
      result.cause := loader.io.done.bits.cause
      result.tval  := loader.io.done.bits.tval
    }
    imageEntry(buffer) := loader.io.done.bits.entry
    imageBytes(buffer)     := loader.io.done.bits.textBytes
    imageConstants(buffer) := loader.io.done.bits.constBytes
    state                  := completed
  }
  when(state === descriptorSend && ports(0).io.request.fire)(state := descriptorWait)
  when(state === descriptorWait && ports(0).io.response.fire) {
    when(ports(0).io.response.bits.error) {
      result.fault := true.B
      result.cause := 5.U
      result.tval  := (command.read_groups.pad(64) << 32) + command.cmd.rs2 + word * 8.U
      state        := completed
    }.otherwise {
      descriptor(word) := ports(0).io.response.bits.data
      word             := word + 1.U
      state            := Mux(word === 11.U, launch, descriptorSend)
    }
  }
  when(state === launch) {
    when(descriptor(0) =/= imageEntry(buffer) || descriptor(1) =/= imageBytes(buffer) || descriptor(
      11
    ) =/= 0.U || descriptor(2) =/= "h40002000".U) {
      execution.io.launch.valid := false.B
      result.fault              := true.B
      result.cause              := 1.U
      result.tval               := descriptor(0)
      state                     := completed
    }.elsewhen(execution.io.launch.fire)(state := executing)
  }
  when(execution.io.done.fire) {
    result := execution.io.done.bits
    state  := completed
  }
  when(io.cmdResp.fire) {
    when(command.cmd.funct7 === 44.U && !result.fault && terminalSeen && !terminalFailed) {
      validImage(buffer) := true.B
    }
    state := idle
  }
}
