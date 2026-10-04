package framework.system.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.memory.cpu.{CpuMemParams, UncachedRequest, UncachedResponse}
import memcore.memory.uncached_ram.{Params => RamParams, Ram}
import memcore.memory.ddr.{Params => DdrParams, Bridge}

/** Chip-wide DDR ordering point; device traffic has a separate, explicit endpoint. */
@instantiable
class Memory(ram: RamParams, ddr: DdrParams, dmaMasters: Int = 0) extends Module {
  require(ram.line == ddr.line && ddr.clients == ram.ports && ddr.slotsPerClient >= ram.slotsPerPort)
  require(dmaMasters >= 0)
  private val orderingParams = ram.copy(externalPorts = 2 * dmaMasters)
  private val cpu            = CpuMemParams(ram.line, ram.tagBits)

  @public val io = IO(new Bundle {
    val cpuRequest     = Vec(ram.cpus, Flipped(Decoupled(new UncachedRequest(cpu))))
    val cpuResponse    = Vec(ram.cpus, Decoupled(new UncachedResponse(cpu)))
    val deviceRequest  = Vec(ram.cpus, Decoupled(new UncachedRequest(cpu)))
    val deviceResponse = Vec(ram.cpus, Flipped(Decoupled(new UncachedResponse(cpu))))
    // CPU L2 backing clients use the CPU memory ordering path.
    val lineRequest    = Vec(ram.lineAgents, Flipped(Decoupled(new LineRequest(ram.line))))
    val lineResponse   = Vec(ram.lineAgents, Decoupled(new LineResponse(ram.line)))
    val dma            = Vec(dmaMasters, Flipped(new memcore.bus.axi4.Port(ddr.axi)))
    val axi            = new memcore.bus.axi4.Port(ddr.axi)
    val outstanding    = Output(UInt(log2Ceil(ram.slots + 2 * dmaMasters + 1).W))
  })

  val ordering       = Instantiate(new Ram(orderingParams))
  val bridge         = Instantiate(new Bridge(ddr))
  ordering.io.lineRequest <> io.lineRequest
  io.lineResponse <> ordering.io.lineResponse
  bridge.io.request <> ordering.io.memoryRequest
  ordering.io.memoryResponse <> bridge.io.response
  val fabric         = Instantiate(new memcore.bus.axi4.Interconnect(ddr.axi, dmaMasters + 1))
  fabric.io.in(0) <> bridge.io.axi
  io.axi <> fabric.io.out
  val dmaOutstanding = Wire(Vec(2 * dmaMasters, Bool()))
  for (i <- 0 until dmaMasters) {
    val source  = io.dma(i)
    val target  = fabric.io.in(i + 1)
    val reading = RegInit(false.B)
    val writing = RegInit(false.B)
    dmaOutstanding(2 * i)     := reading
    dmaOutstanding(2 * i + 1) := writing
    val readAddress  = Reg(UInt(ddr.axi.addressBits.W))
    val writeAddress = Reg(UInt(ddr.axi.addressBits.W))
    val readBytes    = Reg(UInt(13.W))
    val writeBytes   = Reg(UInt(13.W))
    val readRange    = ordering.io.external(2 * i)
    val writeRange   = ordering.io.external(2 * i + 1)
    readRange.valid                                   := reading || source.ar.valid
    readRange.addr                                    := Mux(reading, readAddress, source.ar.bits.addr)
    readRange.bytes                                   := Mux(reading, readBytes, (source.ar.bits.len.pad(13) + 1.U) << source.ar.bits.size)
    readRange.write                                   := false.B
    writeRange.valid                                  := writing || source.aw.valid
    writeRange.addr                                   := Mux(writing, writeAddress, source.aw.bits.addr)
    writeRange.bytes                                  := Mux(writing, writeBytes, (source.aw.bits.len.pad(13) + 1.U) << source.aw.bits.size)
    writeRange.write                                  := true.B
    target.ar.valid                                   := source.ar.valid && !reading && ordering.io.externalAllow(2 * i)
    target.ar.bits                                    := source.ar.bits
    source.ar.ready                                   := target.ar.ready && !reading && ordering.io.externalAllow(2 * i)
    target.aw.valid                                   := source.aw.valid && !writing && ordering.io.externalAllow(2 * i + 1)
    target.aw.bits                                    := source.aw.bits
    source.aw.ready                                   := target.aw.ready && !writing && ordering.io.externalAllow(2 * i + 1)
    target.w <> source.w
    source.r <> target.r
    source.b <> target.b
    when(source.ar.fire) {
      reading := true.B; readAddress := source.ar.bits.addr; readBytes := readRange.bytes
    }
    when(source.r.fire && source.r.bits.last)(reading := false.B)
    when(source.aw.fire) {
      writing := true.B; writeAddress := source.aw.bits.addr; writeBytes := writeRange.bytes
    }
    when(source.b.fire)(writing                       := false.B)
  }
  io.outstanding := ordering.io.outstanding +& (if (dmaMasters == 0) 0.U else PopCount(dmaOutstanding))

  for (i <- 0 until ram.cpus) {
    val request     = io.cpuRequest(i)
    val response    = io.cpuResponse(i)
    val active      = RegInit(false.B)
    val normalOwner = Reg(Bool())
    val tag         = Reg(UInt(ram.tagBits.W))
    val normal      = ordering.io.cpuRequest(i)
    val result      = ordering.io.cpuResponse(i)
    normal.valid               := request.valid && !active && request.bits.normal
    normal.bits.addr           := request.bits.addr
    normal.bits.tag            := request.bits.tag
    normal.bits.size           := request.bits.size
    normal.bits.write          := request.bits.write
    normal.bits.data           := request.bits.data
    normal.bits.atomic         := request.bits.atomic
    io.deviceRequest(i).valid  := request.valid && !active && !request.bits.normal
    io.deviceRequest(i).bits   := request.bits
    request.ready              := !active && Mux(request.bits.normal, normal.ready, io.deviceRequest(i).ready)
    response.valid             := active && Mux(normalOwner, result.valid, io.deviceResponse(i).valid)
    response.bits.tag          := Mux(normalOwner, result.bits.tag, io.deviceResponse(i).bits.tag)
    response.bits.data         := Mux(normalOwner, result.bits.data, io.deviceResponse(i).bits.data)
    response.bits.error        := Mux(normalOwner, result.bits.error, io.deviceResponse(i).bits.error)
    result.ready               := active && normalOwner && response.ready
    io.deviceResponse(i).ready := active && !normalOwner && response.ready
    when(request.fire) {
      active      := true.B
      normalOwner := request.bits.normal
      tag         := request.bits.tag
      assert(request.bits.normal || request.bits.atomic === 0.U, "Device traffic cannot execute RAM atomics")
    }
    when(response.valid) {
      assert(response.bits.tag === tag, "Chip memory response crossed a CPU owner")
    }
    when(response.fire)(active := false.B)
  }
}
