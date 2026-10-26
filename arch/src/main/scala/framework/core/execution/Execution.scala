package framework.core.execution

import chisel3.util._
import framework.acc.frontend.RoCCIO
import framework.core.controller.admission.Commands
import framework.core.clink.CoreInterrupts
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.execution.configs.ExecutionParam
import framework.core.execution.pipeline.{Retirement, Sequencer}
import framework.core.execution.instruction.Instruction
import framework.core.execution.register_file.RegisterFile
import framework.core.execution.integer.Integer
import framework.core.execution.muldiv.MulDiv
import framework.core.execution.fpu.FPU
import framework.core.execution.privileged.Privileged
import framework.core.memory.access.{UncachedRequest, UncachedResponse, VirtualPort}

/** Composes instruction execution; memory requests are served by the core memory hierarchy. */
@instantiable
class Execution(b: ExecutionParam, localRocc: Boolean) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts   = Input(new CoreInterrupts)
    val fetch        = Flipped(new VirtualPort(b.memoryAccess))
    val data         = Flipped(new VirtualPort(b.memoryAccess))
    val timeRequest  = Decoupled(new UncachedRequest(b.memoryAccess.memory))
    val timeResponse = Flipped(Decoupled(new UncachedResponse(b.memoryAccess.memory)))
    val controller   = Option.when(!localRocc)(new Commands(b.tracking, b.integer.xLen, b.privileged.permission))
    val rocc         = Option.when(localRocc)(Flipped(new RoCCIO(b.integer.xLen, b.privileged.permission)))
    val workDrained  = Input(Bool())
    val retirement   = Valid(new Retirement)
  })

  val sequencer:   Instance[Sequencer]    = Instantiate(new Sequencer(b, localRocc))
  val instruction: Instance[Instruction]  = Instantiate(new Instruction(b.instruction))
  val registers:   Instance[RegisterFile] = Instantiate(new RegisterFile(b.registerFile))
  val integer:     Instance[Integer]      = Instantiate(new Integer(b.integer))
  val multiply:    Instance[MulDiv]       = Instantiate(new MulDiv(b.muldiv))
  val floating:    Instance[FPU]          = Instantiate(new FPU(b.fpu))
  val privileged:  Instance[Privileged]   = Instantiate(new Privileged(b.privileged))

  sequencer.io.instruction <> instruction.io
  sequencer.io.registers <> registers.io
  sequencer.io.integer <> integer.io
  sequencer.io.multiply <> multiply.io
  sequencer.io.floating <> floating.io
  sequencer.io.privileged <> privileged.io
  sequencer.io.interrupts  := io.interrupts
  sequencer.io.controller.zip(io.controller).foreach { case (child, parent) => child <> parent }
  sequencer.io.rocc.zip(io.rocc).foreach { case (child, parent) => child <> parent }
  io.retirement            := sequencer.io.retirement
  sequencer.io.workDrained := io.workDrained

  io.fetch <> sequencer.io.fetch
  io.data <> sequencer.io.memory
  io.timeRequest <> sequencer.io.timeRequest
  sequencer.io.timeResponse <> io.timeResponse
}
