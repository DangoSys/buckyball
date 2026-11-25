package framework.hart.execution

import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import framework.generic.contract.memdomain.DmaFaultControl

import chisel3.util._
import framework.generic.contract.controller.{Commands}
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.execution.configs.ExecutionParam
import framework.hart.execution.pipeline.{Retirement, Sequencer}
import framework.hart.execution.instruction.Instruction
import framework.hart.execution.register_file.RegisterFile
import framework.hart.execution.integer.Integer
import framework.hart.execution.muldiv.MulDiv
import framework.hart.execution.fpu.FPU
import framework.hart.execution.privileged.Privileged
import framework.hart.memory.access.VirtualPort
import framework.generic.contract.frontend.RoCCIO
import framework.generic.contract.hlink.HartInterrupts

/** Composes instruction execution; memory requests are served by the hart memory hierarchy. */
@instantiable
class Execution(b: ExecutionParam, localRocc: Boolean) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts   = Input(new HartInterrupts)
    val fetch        = Flipped(new VirtualPort(b.memoryAccess))
    val data         = Flipped(new VirtualPort(b.memoryAccess.copy(lineBits = 0)))
    val timeRequest  = Decoupled(new UncachedRequest(b.memoryAccess.memory))
    val timeResponse = Flipped(Decoupled(new UncachedResponse(b.memoryAccess.memory)))
    val controller   = Option.when(!localRocc)(new Commands(b.tracking, b.integer.xLen, b.privileged.permission))
    val rocc         = Option.when(localRocc)(Flipped(new RoCCIO(b.integer.xLen, b.privileged.permission)))
    val workDrained  = Input(Bool())
    val fault        = Option.when(b.directAccelerator)(Flipped(new DmaFaultControl))
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
  sequencer.io.fault.zip(io.fault).foreach { case (child, parent) => child <> parent }
  io.retirement            := sequencer.io.retirement
  sequencer.io.workDrained := io.workDrained

  io.fetch <> sequencer.io.fetch
  io.data <> sequencer.io.memory
  io.timeRequest <> sequencer.io.timeRequest
  sequencer.io.timeResponse <> io.timeResponse
}
