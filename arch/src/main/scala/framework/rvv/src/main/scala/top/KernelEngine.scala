package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.rvv.configs.RvvParam

@instantiable
class KernelEngine(val p: RvvParam = RvvParam()) extends Module {

  @public
  val io = IO(new Bundle {
    val program = Flipped(Decoupled(new ProgramWrite))
    val data    = Flipped(Decoupled(new DataWrite))
    val launch  = Flipped(Decoupled(new KernelLaunch))
    val done    = Decoupled(new KernelCompletion)
    val busy    = Output(Bool())
  })

  val iBuf:            Seq[Instance[IBuf]]       = Seq.fill(2)(Instantiate(new IBuf(p)))
  val dBuf:            Seq[Instance[DBuf]]       = Seq.fill(2)(Instantiate(new DBuf(p)))
  val scalarRegisters: Instance[ScalarRF]        = Instantiate(new ScalarRF)
  val scalarExecution: Instance[ScalarExecution] = Instantiate(new ScalarExecution)
  val vectorCore:      Instance[VectorCore]      = Instantiate(new VectorCore(p))

  val running          = RegInit(false.B)
  val fetchPending     = RegInit(false.B)
  val instructionValid = RegInit(false.B)
  val waitingVector    = RegInit(false.B)
  val completionValid  = RegInit(false.B)
  val pc               = Reg(UInt(32.W))
  val end              = Reg(UInt(32.W))
  val instruction      = Reg(UInt(32.W))
  val cycles           = RegInit(0.U(64.W))
  val completionFault  = Reg(Bool())
  val completionCause  = Reg(UInt(32.W))
  val completionValue  = Reg(UInt(32.W))
  val activeIBuf       = RegInit(false.B)
  val activeDBuf       = RegInit(false.B)

  val fetch = running && !fetchPending && !instructionValid && !waitingVector && pc < end

  for (buffer <- 0 until 2) {
    iBuf(buffer).io.upload.valid  := io.program.valid && io.program.bits.buffer === buffer.U
    iBuf(buffer).io.upload.bits   := io.program.bits
    iBuf(buffer).io.uploadEnabled := !io.launch.valid && (!running || activeIBuf =/= buffer.U)
    iBuf(buffer).io.fetchEnabled  := fetch && activeIBuf === buffer.U
    iBuf(buffer).io.fetchAddress  := pc

    dBuf(buffer).io.load.valid  := io.data.valid && io.data.bits.buffer === buffer.U
    dBuf(buffer).io.load.bits   := io.data.bits
    dBuf(buffer).io.loadEnabled := !io.launch.valid && (!running || activeDBuf =/= buffer.U)
  }
  io.program.ready := Mux(io.program.bits.buffer, iBuf(1).io.upload.ready, iBuf(0).io.upload.ready)
  io.data.ready := Mux(io.data.bits.buffer, dBuf(1).io.load.ready, dBuf(0).io.load.ready)

  val fetchedInstruction = Mux(activeIBuf, iBuf(1).io.instruction, iBuf(0).io.instruction)
  val fetchedLoaded      = Mux(activeIBuf, iBuf(1).io.loaded, iBuf(0).io.loaded)

  scalarRegisters.io.xReadAddress1    := instruction(19, 15)
  scalarRegisters.io.xReadAddress2    := instruction(24, 20)
  scalarRegisters.io.fReadAddress1    := 0.U
  scalarRegisters.io.fReadAddress2    := 0.U
  scalarRegisters.io.fReadAddress3    := 0.U
  scalarRegisters.io.initialize.valid := io.launch.fire
  scalarRegisters.io.initialize.bits  := io.launch.bits.args
  scalarRegisters.io.xWrite.valid     := false.B
  scalarRegisters.io.xWrite.bits      := DontCare
  scalarRegisters.io.fWrite.valid     := false.B
  scalarRegisters.io.fWrite.bits      := DontCare

  scalarExecution.io.instruction := instruction
  scalarExecution.io.pc          := pc
  scalarExecution.io.source1     := scalarRegisters.io.xReadData1
  scalarExecution.io.source2     := scalarRegisters.io.xReadData2

  val vectorInstruction = instruction(6, 0) === "h57".U ||
    instruction(6, 0) === "h07".U && instruction(14, 12) === 6.U ||
    instruction(6, 0) === "h27".U && instruction(14, 12) === 6.U

  vectorCore.io.issue.valid            := instructionValid && vectorInstruction
  vectorCore.io.issue.bits.instruction := instruction
  vectorCore.io.issue.bits.scalar1     := scalarRegisters.io.xReadData1
  vectorCore.io.issue.bits.scalar2     := scalarRegisters.io.xReadData2
  vectorCore.io.result.ready           := waitingVector

  for (port <- 0 until p.memoryPorts) {
    dBuf(0).io.request(port).valid          := vectorCore.io.memoryRequest(port).valid && !activeDBuf
    dBuf(1).io.request(port).valid          := vectorCore.io.memoryRequest(port).valid && activeDBuf
    dBuf(0).io.request(port).bits           := vectorCore.io.memoryRequest(port).bits
    dBuf(1).io.request(port).bits           := vectorCore.io.memoryRequest(port).bits
    vectorCore.io.memoryRequest(port).ready :=
      Mux(activeDBuf, dBuf(1).io.request(port).ready, dBuf(0).io.request(port).ready)

    vectorCore.io.memoryResponse(port).valid :=
      Mux(activeDBuf, dBuf(1).io.response(port).valid, dBuf(0).io.response(port).valid)
    vectorCore.io.memoryResponse(port).bits  :=
      Mux(activeDBuf, dBuf(1).io.response(port).bits, dBuf(0).io.response(port).bits)
    dBuf(0).io.response(port).ready          := vectorCore.io.memoryResponse(port).ready && !activeDBuf
    dBuf(1).io.response(port).ready          := vectorCore.io.memoryResponse(port).ready && activeDBuf
  }

  io.launch.ready          := !running && !completionValid
  io.done.valid            := completionValid
  io.done.bits.fault       := completionFault
  io.done.bits.pc          := pc
  io.done.bits.instruction := instruction
  io.done.bits.cycles      := cycles
  io.done.bits.cause       := completionCause
  io.done.bits.tval        := completionValue
  io.busy                  := running || completionValid

  when(io.done.fire) {
    completionValid := false.B
  }

  when(io.launch.fire) {
    pc               := io.launch.bits.entry
    end              := io.launch.bits.end
    cycles           := 0.U
    instruction      := 0.U
    fetchPending     := false.B
    instructionValid := false.B
    waitingVector    := false.B
    activeIBuf       := io.launch.bits.iBuffer
    activeDBuf       := io.launch.bits.dBuffer
    when(io.launch.bits.entry(1, 0).orR || io.launch.bits.end(1, 0).orR ||
      io.launch.bits.entry >= io.launch.bits.end || io.launch.bits.end > (p.iBufWords * 4).U) {
      running         := false.B
      completionValid := true.B
      completionFault := true.B
      completionCause := 1.U
      completionValue := io.launch.bits.entry
    }.otherwise {
      running := true.B
    }
  }

  when(running) {
    cycles := cycles + 1.U
  }

  when(fetch) {
    when(fetchedLoaded) {
      fetchPending := true.B
    }.otherwise {
      running         := false.B
      completionValid := true.B
      completionFault := true.B
      completionCause := 1.U
      completionValue := pc
    }
  }

  when(fetchPending) {
    instruction      := fetchedInstruction
    instructionValid := true.B
    fetchPending     := false.B
  }

  when(instructionValid && vectorInstruction && vectorCore.io.issue.fire) {
    instructionValid := false.B
    waitingVector    := true.B
  }

  when(instructionValid && !vectorInstruction) {
    when(scalarExecution.io.legal && !scalarExecution.io.nextPc(1, 0).orR) {
      when(scalarExecution.io.write) {
        scalarRegisters.io.xWrite.valid        := true.B
        scalarRegisters.io.xWrite.bits.address := scalarExecution.io.destination
        scalarRegisters.io.xWrite.bits.data    := scalarExecution.io.result
      }
      pc               := scalarExecution.io.nextPc
      instructionValid := false.B
    }.otherwise {
      running          := false.B
      instructionValid := false.B
      completionValid  := true.B
      completionFault  := true.B
      completionCause  := Mux(scalarExecution.io.legal, 0.U, 2.U)
      completionValue  := Mux(scalarExecution.io.legal, scalarExecution.io.nextPc, instruction)
    }
  }

  when(vectorCore.io.result.fire) {
    waitingVector := false.B
    when(vectorCore.io.result.bits.fault) {
      running         := false.B
      completionValid := true.B
      completionFault := true.B
      completionCause := vectorCore.io.result.bits.cause
      completionValue := vectorCore.io.result.bits.tval
    }.otherwise {
      when(vectorCore.io.result.bits.scalarWrite) {
        scalarRegisters.io.xWrite.valid        := true.B
        scalarRegisters.io.xWrite.bits.address := instruction(11, 7)
        scalarRegisters.io.xWrite.bits.data    := vectorCore.io.result.bits.scalarData
      }
      pc := pc + 4.U
    }
  }

  when(running && pc === end && !fetchPending && !instructionValid && !waitingVector) {
    running         := false.B
    completionValid := true.B
    completionFault := false.B
    completionCause := 0.U
    completionValue := 0.U
  }
}

object EmitKernelEngine extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new KernelEngine(),
    args,
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
