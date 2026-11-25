package examples.designs.goban.harts.goban

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.rocket.Execution
import examples.hart.rocket.configs.{Params => RocketParam}
import framework.hart.configs.HartParam
import framework.hart.memory.{MemCtrl, Memory}
import framework.generic.contract.controller.{Commands}
import framework.generic.bus.{axi4, chi}
import framework.generic.contract.hlink.HartInterrupts

/** Outer Loom execution hart, implemented by Rocket; runs the kernel that schedules task threads. */
@instantiable
class Loom(hartp: HartParam, rocketp: RocketParam) extends Module {
  require(!hartp.hlink.backing)

  @public
  val io = IO(new Bundle {
    val interrupts    = Input(new HartInterrupts)
    val commands      = new Commands(hartp.memory.dcache.tracking, hartp.integer.xLen, hartp.privileged.permission)
    val memoryAllowed = Input(Bool())
    val memory        =
      Flipped(new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker))
    val coherent      = Vec(2, new chi.RequesterPort(hartp.memory.home.memory.chi))
    val uncached      = new axi4.Port(hartp.memory.uncached.axi)
  })

  val execution: Instance[Execution] = Instantiate(new Execution(rocketp, commandsEnabled = true))
  val memory:    Instance[Memory]    = Instantiate(new Memory(hartp.memory, externalHome = true))
  execution.io.interrupts  := io.interrupts
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.probe.get <> execution.io.probe
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := io.memoryAllowed
  memory.io.dma.get <> io.memory
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
  // Thread tasks outlive their management commands; the kernel must keep running.
  rocc.busy                               := pending
  rocc.interrupt                          := false.B
  // Goban's CPU has no directly attached accelerator; thread DMA faults reach it as tasks.
  execution.io.fault.fault.valid          := false.B
  execution.io.fault.fault.bits           := 0.U.asTypeOf(execution.io.fault.fault.bits)
}
