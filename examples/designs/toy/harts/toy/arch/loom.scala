package examples.designs.toy.harts.toy

import framework.generic.contract.memdomain.DmaFaultControl

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.rocket.configs.{Params => RocketParam}
import examples.hart.rocket.Execution
import framework.hart.memory.{MemCtrl, MemorySystem}
import framework.generic.bus.axi4
import framework.generic.bus.chi.snf.{LineRequest, LineResponse}
import framework.generic.contract.frontend.RoCCIO
import framework.generic.contract.hlink.HartInterrupts

/** Outer Loom execution hart and private memory hierarchy; directly controls the local ACC. */
@instantiable
class Loom(hartp: RocketParam) extends Module {
  require(hartp.hlink.backing)

  @public
  val io = IO(new Bundle {
    val interrupts = Input(new HartInterrupts)
    val rocc       = Flipped(new RoCCIO(hartp.cpu.core.xLen, hartp.permission))
    val memory     = Flipped(new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker))
    val fault      = Flipped(new DmaFaultControl)
    val uncached   = new axi4.Port(hartp.memory.uncached.axi)

    val backing = new Bundle {
      val request  = Decoupled(new LineRequest(hartp.memory.home.memory.chi))
      val response = Flipped(Decoupled(new LineResponse(hartp.memory.home.memory.chi)))
    }

  })

  val execution: Instance[Execution] = Instantiate(new Execution(hartp, commandsEnabled = true))

  val memory: Instance[MemorySystem] = Instantiate(new MemorySystem(hartp.memory, externalHome = false))

  execution.io.interrupts  := io.interrupts
  execution.io.rocc <> io.rocc
  execution.io.fault <> io.fault
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.probe.get <> execution.io.probe
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed := !io.rocc.busy
  memory.io.services <> io.memory
  io.uncached <> memory.io.uncached
  io.backing <> memory.io.backing.get

}
