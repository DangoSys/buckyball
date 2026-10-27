package examples.core.reduce

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.execution.Execution
import framework.core.memory.{MemoryPort, MemorySystem}
import framework.core.controller.admission.Commands
import framework.core.clink.CoreInterrupts
import framework.ip.bus.{axi4, chi}

@instantiable
class Loom(corep: CoreParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts = Input(new CoreInterrupts)
    val commands   = new Commands(corep.memory.dcache.tracking, corep.integer.xLen, corep.privileged.permission)
    val drained    = Input(Bool())
    val memory     = Flipped(new MemoryPort(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker))
    val coherent   = Vec(2, new chi.RequesterPort(corep.memory.home.memory.chi))
    val uncached   = new axi4.Port(corep.memory.uncached.axi)
  })

  val execution: Instance[Execution]    = Instantiate(new Execution(corep.execution(directAccelerator = true), false))
  val memory:    Instance[MemorySystem] = Instantiate(new MemorySystem(corep.memory, externalHome = true))
  execution.io.interrupts  := io.interrupts
  io.commands <> execution.io.controller.get
  execution.io.workDrained := io.drained
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := io.drained
  memory.io.services <> io.memory
  io.uncached <> memory.io.uncached
  io.coherent <> memory.io.coherent
}
