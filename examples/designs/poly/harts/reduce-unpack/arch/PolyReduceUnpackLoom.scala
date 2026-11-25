package examples.designs.poly.harts.reduceunpack

import framework.generic.contract.memdomain.DmaFaultControl

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.execution.Execution
import framework.hart.memory.{MemCtrl, MemorySystem}
import framework.generic.contract.controller.{Commands}
import framework.generic.bus.{axi4, chi}
import framework.generic.contract.hlink.HartInterrupts

@instantiable
class PolyReduceUnpackLoom(hartp: HartParam) extends Module {

  @public
  val io = IO(new Bundle {
    val interrupts    = Input(new HartInterrupts)
    val commands      = new Commands(hartp.memory.dcache.tracking, hartp.integer.xLen, hartp.privileged.permission)
    val drained       = Input(Bool())
    val memory        = Flipped(new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker))
    val payload       = Flipped(new MemCtrl(hartp.memory.dcache.tracking, hartp.memory.memoryAccess.walker))
    val payloadActive = Input(Bool())
    val fault         = Flipped(new DmaFaultControl)
    val coherent      = Vec(2, new chi.RequesterPort(hartp.memory.home.memory.chi))
    val uncached      = new axi4.Port(hartp.memory.uncached.axi)
  })

  val execution: Instance[Execution]    = Instantiate(new Execution(hartp.execution(directAccelerator = true), false))
  val memory:    Instance[MemorySystem] = Instantiate(new MemorySystem(hartp.memory, externalHome = true))
  execution.io.interrupts      := io.interrupts
  io.commands <> execution.io.controller.get
  execution.io.workDrained     := io.drained
  execution.io.fault.get <> io.fault
  memory.io.fetch <> execution.io.fetch
  memory.io.memory <> execution.io.data
  memory.io.timeRequest <> execution.io.timeRequest
  execution.io.timeResponse <> memory.io.timeResponse
  memory.io.commandAllowed     := io.drained
  memory.io.services <> io.memory
  io.payload.cpuQuery          := memory.io.services.cpuQuery
  io.payload.lineMaintain      := memory.io.services.lineMaintain
  io.payload.drained           := memory.io.services.drained
  io.payload.maintenance.ready := false.B
  io.payload.maintained.valid  := false.B
  io.payload.maintained.bits   := memory.io.services.maintained.bits
  io.payload.pteRequest.ready  := false.B
  io.payload.pteResponse.valid := false.B
  io.payload.pteResponse.bits  := memory.io.services.pteResponse.bits
  when(io.payloadActive) {
    memory.io.services <> io.payload
    io.memory.maintenance.ready := false.B
    io.memory.maintained.valid  := false.B
    io.memory.pteRequest.ready  := false.B
    io.memory.pteResponse.valid := false.B
  }
  io.uncached <> memory.io.uncached
  io.coherent <> memory.io.coherent
}
