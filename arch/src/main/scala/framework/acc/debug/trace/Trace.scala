package framework.acc.debug.trace

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.ant.Retire

class InstructionEvent extends Bundle {
  val is_issue  = UInt(8.W)
  val thread_id = UInt(64.W)
  val rob_id    = UInt(32.W)
  val inst_id   = UInt(64.W)
  val domain_id = UInt(32.W)
  val funct     = UInt(32.W)
  val pc        = UInt(64.W)
  val rs1_data  = UInt(64.W)
  val rs2_data  = UInt(64.W)
  val enable    = Bool()
}

class MemoryEvent extends Bundle {
  val is_write   = UInt(8.W)
  val is_shared  = UInt(8.W)
  val channel    = UInt(32.W)
  val thread_id  = UInt(64.W)
  val rob_id     = UInt(32.W)
  val inst_id    = UInt(64.W)
  val vbank_id   = UInt(32.W)
  val pbank_id   = UInt(32.W)
  val group_id   = UInt(32.W)
  val addr       = UInt(32.W)
  val write_mask = UInt(32.W)
  val data_lo    = UInt(64.W)
  val data_hi    = UInt(64.W)
  val enable     = Bool()
}

class AcceleratorEvents(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  gpParam:       GpDomainParam,
  ballParam:     BallDomainParam)
    extends Bundle {

  val memoryEvents  = memParam.bankRequestPorts +
    (if (gpParam.enable) gpParam.memory.memoryPorts else 0) + 2 + memParam.bankNum
  val memory        = Vec(memoryEvents, new MemoryEvent)
  val ballStarted   = Vec(ballParam.balls.size, Valid(UInt(log2Ceil(frontendParam.rob_entries).W)))
  val ballCompleted = Vec(ballParam.balls.size, Valid(UInt(log2Ceil(frontendParam.rob_entries).W)))
}

/** Owns trace sinks and performance counters for one hardware thread. */
@instantiable
class Trace(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  gpParam:       GpDomainParam,
  ballParam:     BallDomainParam,
  threadId:      Int)
    extends Module {

  @public
  val io = IO(new Bundle {
    val instructions = Input(Vec(4, new InstructionEvent))
    val ant          = Input(Valid(new Retire))
    val acc          = Input(new AcceleratorEvents(frontendParam, memParam, gpParam, ballParam))
  })

  val ant: Instance[AntTraceDPI] = Instantiate(new AntTraceDPI)
  ant.io.clock       := clock
  ant.io.reset       := reset.asBool
  ant.io.thread_id   := threadId.U(64.W)
  ant.io.pc          := io.ant.bits.pc
  ant.io.instruction := io.ant.bits.instruction
  ant.io.rd          := io.ant.bits.rd
  ant.io.data        := io.ant.bits.data
  ant.io.enable      := io.ant.valid

  for (event <- io.instructions) {
    val sink: Instance[ITraceDPI] = Instantiate(new ITraceDPI)
    sink.io.clock     := clock
    sink.io.reset     := reset.asBool
    sink.io.is_issue  := event.is_issue
    sink.io.thread_id := threadId.U(64.W)
    sink.io.rob_id    := event.rob_id
    sink.io.inst_id   := event.inst_id
    sink.io.domain_id := event.domain_id
    sink.io.funct     := event.funct
    sink.io.pc        := event.pc
    sink.io.rs1_data  := event.rs1_data
    sink.io.rs2_data  := event.rs2_data
    sink.io.enable    := event.enable
  }
  for (event <- io.acc.memory) {
    val sink: Instance[MTraceDPI] = Instantiate(new MTraceDPI)
    sink.io.clock      := clock
    sink.io.reset      := reset.asBool
    sink.io.is_write   := event.is_write
    sink.io.is_shared  := event.is_shared
    sink.io.channel    := event.channel
    sink.io.thread_id  := threadId.U(64.W)
    sink.io.rob_id     := event.rob_id
    sink.io.inst_id    := event.inst_id
    sink.io.vbank_id   := event.vbank_id
    sink.io.pbank_id   := event.pbank_id
    sink.io.group_id   := event.group_id
    sink.io.addr       := event.addr
    sink.io.write_mask := event.write_mask
    sink.io.data_lo    := event.data_lo
    sink.io.data_hi    := event.data_hi
    sink.io.enable     := event.enable
  }
  val cycle = RegInit(0.U(64.W))
  cycle := cycle + 1.U
  val starts = Reg(Vec(io.acc.ballStarted.size, UInt(64.W)))
  for (index <- io.acc.ballStarted.indices) {
    when(io.acc.ballStarted(index).valid)(starts(index) := cycle)
    val sink = Instantiate(new PMCTraceDPI)
    sink.io.clock   := clock
    sink.io.reset   := reset.asBool
    sink.io.ball_id := index.U
    sink.io.rob_id  := io.acc.ballCompleted(index).bits
    sink.io.elapsed := cycle - starts(index)
    sink.io.enable  := io.acc.ballCompleted(index).valid
  }
}
