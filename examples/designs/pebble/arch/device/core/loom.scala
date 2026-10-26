package examples.designs.pebble

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.configs.CoreParam
import framework.core.execution.Execution
import framework.core.memory.{MemoryPort, MemorySystem}
import framework.acc.frontend.RoCCIO
import framework.core.clink.CoreInterrupts
import framework.ip.bus.axi4
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

/** Outer Loom execution core and private memory hierarchy; directly controls the local ACC. */
@instantiable
class Loom(corep: CoreParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts = Input(new CoreInterrupts)
    val rocc       = Flipped(new RoCCIO(corep.integer.xLen, corep.privileged.permission))
    val memory     = Flipped(new MemoryPort(corep.memory.dcache.tracking, corep.memory.memoryAccess.walker))
    val uncached   = new axi4.Port(corep.memory.uncached.axi)

    val backing = new Bundle {
      val request  = Decoupled(new LineRequest(corep.memory.home.memory.chi))
      val response = Flipped(Decoupled(new LineResponse(corep.memory.home.memory.chi)))
    }

  })

  val execution: Instance[Execution] = Instantiate(new Execution(
    corep.execution(directAccelerator = true),
    localRocc = true
  ))

  val memory: Instance[MemorySystem] = Instantiate(new MemorySystem(corep.memory, externalHome = false))

  execution.io.interrupts  := io.interrupts
  execution.io.rocc.get <> io.rocc
  execution.io.workDrained := !io.rocc.busy
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := !io.rocc.busy
  memory.io.services <> io.memory
  io.uncached <> memory.io.uncached
  io.backing <> memory.io.backing.get

}
