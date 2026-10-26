package examples.designs.goban

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.rocket.Execution
import examples.core.rocket.configs.{Params => RocketParam}
import framework.core.configs.CoreParam
import framework.core.memory.{Memory, MemoryPort}
import framework.core.controller.admission.Commands
import framework.core.clink.CoreInterrupts
import framework.ip.bus.{axi4, chi}

/** Outer Loom execution core, implemented by Rocket; runs the engine that schedules task threads. */
@instantiable
class Loom(corep: CoreParam, rocketp: RocketParam) extends Module {
  require(!corep.clink.backing)

  @public
  val io = IO(new Bundle {
    val interrupts    = Input(new CoreInterrupts)
    val commands      = new Commands(corep.memory.dcache.tracking, corep.integer.xLen, corep.privileged.permission)
    val drained       = Input(Bool())
    val memoryAllowed = Input(Bool())
    val memory        =
      Vec(corep.threads.size, Flipped(new MemoryPort(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker)))
    val coherent      = Vec(2, new chi.RequesterPort(corep.memory.home.memory.chi))
    val uncached      = new axi4.Port(corep.memory.uncached.axi)
  })

  val execution: Instance[Execution] = Instantiate(new Execution(rocketp, commandsEnabled = true))
  val memory:    Instance[Memory]    = Instantiate(new Memory(corep.memory, externalHome = true))
  execution.io.interrupts  := io.interrupts
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := io.memoryAllowed
  memory.io.threads.get <> io.memory
  io.coherent <> memory.io.coherent
  io.uncached <> memory.io.uncached

  val rocc    = execution.io.rocc
  val command = io.commands.command
  val pending = RegInit(false.B)
  command.valid                           := rocc.cmd.valid && !pending
  rocc.cmd.ready                          := command.ready && !pending
  command.bits.tag                        := 0.U
  command.bits.instruction                := rocc.cmd.bits.instruction
  command.bits.satp                       := rocc.cmd.bits.satp
  command.bits.effectivePrivilege         := rocc.cmd.bits.effectivePrivilege
  command.bits.sum                        := rocc.cmd.bits.sum
  command.bits.mxr                        := rocc.cmd.bits.mxr
  command.bits.pmp                        := rocc.cmd.bits.pmp
  when(command.fire)(pending              := true.B)
  io.commands.complete.ready              := true.B
  when(io.commands.complete.fire)(pending := false.B)
  rocc.resp <> io.commands.response
  rocc.busy                               := pending || !io.drained
  rocc.interrupt                          := false.B
}
