package memcore.memory.uncached_ram

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.ddr.{Bridge, Params => DdrParams}

/** Verification composition with the actual AXI bridge, not an early-ack line model. */
@instantiable
class Ddr(p: Params) extends Module {
  override def desiredName = "RamDdr"
  val ddr                  = DdrParams(line = p.line, clients = 2, slotsPerClient = p.slotsPerPort, dataBits = 128, idBits = 4)

  @public
  val io = IO(new Bundle {
    val cpuRequest       = Vec(p.cpus, Flipped(Decoupled(new Request(p))))
    val cpuResponse      = Vec(p.cpus, Decoupled(new Response(p)))
    val lineRequest      = Vec(p.lineAgents, Flipped(Decoupled(new LineRequest(p.line))))
    val lineResponse     = Vec(p.lineAgents, Decoupled(new LineResponse(p.line)))
    val axi              = new memcore.bus.axi4.Port(ddr.axi)
    val observedRequest  = Output(Vec(2, Valid(new LineRequest(p.line))))
    val observedResponse = Output(Vec(2, Valid(new LineResponse(p.line))))
    val outstanding      = Output(UInt(log2Ceil(p.slots + 1).W))
  })

  val ram: Instance[Ram] = Instantiate(new Ram(p)); val bridge: Instance[Bridge] = Instantiate(new Bridge(ddr))
  ram.io.cpuRequest <> io.cpuRequest; io.cpuResponse <> ram.io.cpuResponse
  ram.io.lineRequest <> io.lineRequest; io.lineResponse <> ram.io.lineResponse
  bridge.io.request <> ram.io.memoryRequest; ram.io.memoryResponse <> bridge.io.response
  io.axi <> bridge.io.axi; io.outstanding := ram.io.outstanding
  for (i <- 0 until 2) {
    io.observedRequest(i).valid  := ram.io.memoryRequest(i).fire;
    io.observedRequest(i).bits   := ram.io.memoryRequest(i).bits
    io.observedResponse(i).valid := ram.io.memoryResponse(i).fire;
    io.observedResponse(i).bits  := ram.io.memoryResponse(i).bits
  }
}
