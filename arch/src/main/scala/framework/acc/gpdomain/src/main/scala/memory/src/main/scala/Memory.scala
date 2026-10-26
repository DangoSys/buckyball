package framework.acc.gpdomain.memory

import framework.lib.queue.Queue

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.{BankPort => BankAccessPort}
import framework.acc.gpdomain.execution.{VectorMemoryRequest, VectorMemoryResponse}
import framework.acc.gpdomain.loader.LoaderMemoryPort

class ResponseOwner extends Bundle {
  val local  = Bool()
  val loader = Bool()
}

@instantiable
class Memory(val frontendParam: FrontendParam, val memParam: MemDomainParam, val gpParam: GpDomainParam)
    extends Module {
  val ownerId = frontendParam.ownerId
  val p       = gpParam.memory

  @public
  val io = IO(new Bundle {
    val loader   = Flipped(new LoaderMemoryPort(frontendParam, memParam))
    val request  = Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryRequest)))
    val response = Vec(p.memoryPorts, Decoupled(new VectorMemoryResponse))
    val banks    = Vec(p.memoryPorts, new BankAccessPort(frontendParam, memParam))
  })

  val local: Instance[KernelMemory]  = Instantiate(new KernelMemory(gpParam))
  val ports: Seq[Instance[BankPort]] =
    Seq.fill(p.memoryPorts)(Instantiate(new BankPort(frontendParam, memParam)))

  local.io.buffer     := io.loader.buffer
  local.io.loading    := io.loader.loading
  local.io.constBytes := io.loader.constBytes

  for (i <- 0 until p.memoryPorts) {
    io.banks(i) <> ports(i).io.bank
    val owners           = Instantiate(new Queue(
      new ResponseOwner,
      2
    ))
    val fromLoader       = if (i == 0) io.loader.control else false.B
    val request          = if (i == 0) Mux(fromLoader, io.loader.request.bits, io.request(i).bits) else io.request(i).bits
    val offered          =
      if (i == 0) Mux(fromLoader, io.loader.request.valid, io.request(i).valid)
      else !io.loader.control && io.request(i).valid
    val internal         = request.address >= "h40000000".U && request.address < "h40002000".U
    val destinationReady = Mux(internal, local.io.request(i).ready, ports(i).io.request.ready)

    local.io.request(i).valid           := offered && internal && owners.io.enq.ready
    local.io.request(i).bits            := request
    ports(i).io.request.valid           := offered && !internal && owners.io.enq.ready
    ports(i).io.request.bits.operation  := request
    ports(i).io.request.bits.context    := io.loader.context
    owners.io.enq.valid                 := offered && destinationReady
    owners.io.enq.bits.local            := internal
    owners.io.enq.bits.loader           := fromLoader
    io.request(i).ready                 := !io.loader.control && owners.io.enq.ready && destinationReady
    if (i == 0) io.loader.request.ready := io.loader.control && owners.io.enq.ready && destinationReady

    // Hold response ownership until acceptance, even when the request source changes.
    val responseValid = Mux(owners.io.deq.bits.local, local.io.response(i).valid, ports(i).io.response.valid)
    val responseBits  = Mux(owners.io.deq.bits.local, local.io.response(i).bits, ports(i).io.response.bits)
    val responseReady = Mux(owners.io.deq.bits.loader, io.loader.response.ready, io.response(i).ready)
    io.response(i).valid       := owners.io.deq.valid && !owners.io.deq.bits.loader && responseValid
    io.response(i).bits        := responseBits
    if (i == 0) {
      io.loader.response.valid := owners.io.deq.valid && owners.io.deq.bits.loader && responseValid
      io.loader.response.bits  := responseBits
    }
    local.io.response(i).ready := owners.io.deq.valid && owners.io.deq.bits.local && responseReady
    ports(i).io.response.ready := owners.io.deq.valid && !owners.io.deq.bits.local && responseReady
    owners.io.deq.ready        := responseValid && responseReady
  }
}
