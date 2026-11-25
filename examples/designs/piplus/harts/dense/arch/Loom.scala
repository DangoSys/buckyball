package examples.designs.piplus.harts.dense

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.execution.Execution
import framework.hart.memory.{MemCtrl, Memory}
import framework.generic.contract.controller.{Commands}
import framework.generic.bus.{axi4, chi}
import framework.generic.contract.hlink.HartInterrupts

/** The piplus CPU: framework execution and private caches; hart commands go to the hart controller. */
@instantiable
class Loom(hartp: HartParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts    = Input(new HartInterrupts)
    val commands      = new Commands(hartp.memory.dcache.tracking, hartp.integer.xLen, hartp.privileged.permission)
    val drained       = Input(Bool())
    val memoryAllowed = Input(Bool())

    val dma = Option.when(hartp.threads.nonEmpty)(
      Flipped(new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker))
    )

    val coherent = Vec(2, new chi.RequesterPort(hartp.memory.memoryAccess.memory.chi))
    val uncached = new axi4.Port(hartp.memory.uncached.axi)
  })

  val execution: Instance[Execution] =
    Instantiate(new Execution(hartp.execution(directAccelerator = false), localRocc = false))
  val memory:    Instance[Memory]    = Instantiate(new Memory(hartp.memory, externalHome = true))

  execution.io.interrupts  := io.interrupts
  execution.io.workDrained := io.drained
  io.commands <> execution.io.controller.get
  memory.io.fetch <> execution.io.fetch
  memory.io.data <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := io.memoryAllowed
  io.dma.zip(memory.io.dma).foreach { case (port, service) => port <> service }
  io.coherent <> memory.io.coherent
  io.uncached <> memory.io.uncached
}
