package examples.designs.piplus

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.execution.Execution
import framework.core.memory.{MemCtrl, Memory}
import framework.core.controller.admission.Commands
import framework.core.clink.CoreInterrupts
import framework.ip.bus.{axi4, chi}

/** The piplus CPU: framework execution and private caches; core commands go to the core controller. */
@instantiable
class Loom(corep: CoreParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts    = Input(new CoreInterrupts)
    val commands      = new Commands(corep.memory.dcache.tracking, corep.integer.xLen, corep.privileged.permission)
    val drained       = Input(Bool())
    val memoryAllowed = Input(Bool())

    val threads = Option.when(corep.threads.nonEmpty)(
      Vec(corep.threads.size, Flipped(new MemCtrl(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker)))
    )

    val coherent = Vec(2, new chi.RequesterPort(corep.memory.memoryAccess.memory.chi))
    val uncached = new axi4.Port(corep.memory.uncached.axi)
  })

  val execution: Instance[Execution] =
    Instantiate(new Execution(corep.execution(directAccelerator = false), localRocc = false))
  val memory:    Instance[Memory]    = Instantiate(new Memory(corep.memory, externalHome = true))

  execution.io.interrupts  := io.interrupts
  execution.io.workDrained := io.drained
  io.commands <> execution.io.controller.get
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := io.memoryAllowed
  io.threads.zip(memory.io.threads).foreach { case (port, service) => port <> service }
  io.coherent <> memory.io.coherent
  io.uncached <> memory.io.uncached
}
