package examples.designs.piplus.harts.control

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.execution.Execution
import framework.hart.memory.{MemorySystem}
import framework.generic.bus.{axi4, chi}
import framework.generic.contract.hlink.HartInterrupts

/** The piplus CPU of the control hart: framework execution and private caches, nothing on its RoCC port. */
@instantiable
class Loom(hartp: HartParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts = Input(new HartInterrupts)
    val coherent   = Vec(2, new chi.RequesterPort(hartp.memory.home.memory.chi))
    val uncached   = new axi4.Port(hartp.memory.uncached.axi)
  })

  val execution: Instance[Execution]    =
    Instantiate(new Execution(hartp.execution(directAccelerator = false), localRocc = true))
  val memory:    Instance[MemorySystem] = Instantiate(new MemorySystem(hartp.memory, externalHome = true))
  execution.io.interrupts := io.interrupts
  // The hart has no accelerator and no threads: its custom opcodes are illegal, so no command reaches the port.
  val rocc = execution.io.rocc.get
  rocc.cmd.ready                       := false.B
  rocc.resp.valid                      := false.B
  rocc.resp.bits                       := 0.U.asTypeOf(rocc.resp.bits)
  rocc.busy                            := false.B
  rocc.interrupt                       := false.B
  execution.io.workDrained             := true.B
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed             := true.B
  memory.io.services.cpuAllow          := true.B
  memory.io.services.maintenance.valid := false.B
  memory.io.services.maintenance.bits  := 0.U.asTypeOf(memory.io.services.maintenance.bits)
  memory.io.services.pteRequest.valid  := false.B
  memory.io.services.pteRequest.bits   := 0.U.asTypeOf(memory.io.services.pteRequest.bits)
  memory.io.services.maintained.ready  := true.B
  memory.io.services.lineAddress       := 0.U
  memory.io.services.lineWrite         := false.B
  memory.io.services.pteResponse.ready := true.B
  io.uncached <> memory.io.uncached
  io.coherent <> memory.io.coherent
}
