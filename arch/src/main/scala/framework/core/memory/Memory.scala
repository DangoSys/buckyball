package framework.core.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.memory.configs.MemoryParam
import framework.core.memory.access.{UncachedRequest, UncachedResponse, VirtualPort}
import framework.ip.bus.axi4
import framework.ip.bus.chi.RequesterPort
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

/** Core memory hierarchy and shared cache services for attached threads. */
@instantiable
class Memory(b: MemoryParam, externalHome: Boolean) extends Module {
  val core = b

  @public
  val io = IO(new Bundle {
    val fetch        = new VirtualPort(b.memoryAccess)
    val data         = new VirtualPort(b.memoryAccess)
    val timeRequest  = Flipped(Decoupled(new UncachedRequest(b.memoryAccess.memory)))
    val timeResponse = Decoupled(new UncachedResponse(b.memoryAccess.memory))

    val threads = Option.when(core.threadPorts > 0)(Vec(
      core.threadPorts,
      Flipped(new MemoryPort(b.dcache.tracking, b.memoryAccess.walker))
    ))

    val commandAllowed = Input(Bool())
    val uncached       = new axi4.Port(b.uncached.axi)

    val coherent = Vec(if (externalHome) 2 else 0, new RequesterPort(core.memoryAccess.memory.chi))

    val backing = Option.when(!externalHome)(new Bundle {
      val request  = Decoupled(new LineRequest(core.home.memory.chi))
      val response = Flipped(Decoupled(new LineResponse(core.home.memory.chi)))
    })

  })

  val system: Instance[MemorySystem] = Instantiate(new MemorySystem(b, externalHome))
  system.io.fetch <> io.fetch
  system.io.memory <> io.data
  system.io.timeRequest <> io.timeRequest
  io.timeResponse <> system.io.timeResponse
  system.io.commandAllowed := io.commandAllowed
  io.uncached <> system.io.uncached
  io.backing.zip(system.io.backing).foreach { case (port, backing) => port <> backing }
  io.coherent <> system.io.coherent

  if (core.threadPorts > 0) {
    val threads: Instance[ThreadMemory] =
      Instantiate(new ThreadMemory(b.threadPorts, b.dcache.tracking, b.memoryAccess.walker))
    threads.io.clients <> io.threads.get
    system.io.services <> threads.io.system
  } else {
    system.io.services.cpuAllow          := true.B
    system.io.services.maintenance.valid := false.B
    system.io.services.maintenance.bits  := 0.U.asTypeOf(system.io.services.maintenance.bits)
    system.io.services.pteRequest.valid  := false.B
    system.io.services.pteRequest.bits   := 0.U.asTypeOf(system.io.services.pteRequest.bits)
    system.io.services.maintained.ready  := true.B
    system.io.services.pteResponse.ready := true.B
  }
}
