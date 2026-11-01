package framework.core.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.bus.{axi, axi4}
import framework.thread.tlink.{Address, Dma, Receive, Send}

/** A core-side endpoint selects a compiled edge or the ordinary memory path. */
@instantiable
class Endpoint(p: MeshSharedMemParams, bus: axi4.Params, thread: Int) extends Module {

  @public val io = IO(new Bundle {
    val thread = Flipped(new Dma(bus))
    val task   = Input(UInt(64.W))
    val flows  = Input(Vec(p.bankCount, new Flow(p)))
    val read   = Flipped(new MeshChannel(p))
    val write  = Flipped(new MeshChannel(p))
    val ddr    = new axi4.Port(bus)
    val busy   = Output(Vec(p.bankCount, Bool()))
  })

  val rIdle :: rSelect :: rExternal :: rLocal :: Nil              = Enum(4)
  val wIdle :: wSelect :: wExternal :: wLocal :: wComplete :: Nil = Enum(5)
  val readState                                                   = RegInit(rIdle)
  val writeState                                                  = RegInit(wIdle)
  val read                                                        = Reg(new Address(bus))
  val write                                                       = Reg(new Address(bus))
  val readBank                                                    = Reg(UInt(p.bankBits.W))
  val writeBank                                                   = Reg(UInt(p.bankBits.W))
  val readRow                                                     = Reg(UInt(p.addressBits.W))
  val writeRow                                                    = Reg(UInt(p.addressBits.W))
  val reads                                                       = RegInit(0.U(9.W))
  val returned                                                    = RegInit(0.U(9.W))
  val writes                                                      = RegInit(0.U(9.W))
  val committed                                                   = RegInit(0.U(9.W))
  val writeError                                                  = RegInit(false.B)
  val readData                                                    = Instantiate(new Send(new axi.Beat(bus.dataBits, bus.idBits, 0, 2)))
  val writeData                                                   = Instantiate(new Receive(new axi.Beat(bus.dataBits)))
  io.thread.readData <> readData.io.output
  writeData.io.input <> io.thread.writeData
  io.thread.read.ready  := readState === rIdle && !reset.asBool
  io.thread.write.ready := writeState === wIdle && !reset.asBool
  when(io.thread.read.fire) { read := io.thread.read.bits; reads := 0.U; returned := 0.U; readState := rSelect }
  when(io.thread.write.fire) {
    write      := io.thread.write.bits; writes := 0.U; committed := 0.U
    writeError := false.B; writeState          := wSelect
  }

  val readMatches = VecInit(io.flows.map(f =>
    f.live && f.target === thread.U && f.targetTask === io.task &&
      read.virtualAddress >= f.targetAddress && read.virtualAddress < f.targetAddress +& f.bytes
  ))

  val writeMatches = VecInit(io.flows.map(f =>
    f.live && f.source === thread.U && f.sourceTask === io.task &&
      write.virtualAddress >= f.sourceAddress && write.virtualAddress < f.sourceAddress +& f.bytes
  ))

  val readFlow  = io.flows(PriorityEncoder(readMatches))
  val writeFlow = io.flows(PriorityEncoder(writeMatches))
  when(readState === rSelect && readMatches.asUInt.orR) {
    assert(PopCount(readMatches) === 1.U && readFlow.published, "Dataflow input is ambiguous or unpublished")
    assert(
      read.virtualAddress +& (read.beats << log2Ceil(bus.bytes)) <= readFlow.targetAddress +& readFlow.bytes,
      "DMA read crosses a dataflow boundary"
    )
    readBank  := readFlow.bank
    readRow   := (read.virtualAddress - readFlow.targetAddress) >> log2Ceil(bus.bytes)
    readState := rLocal
  }
  when(writeState === wSelect && writeMatches.asUInt.orR) {
    assert(!writeFlow.published, "Dataflow producer wrote a published value")
    assert(
      write.virtualAddress +& (write.beats << log2Ceil(bus.bytes)) <= writeFlow.sourceAddress +& writeFlow.bytes,
      "DMA write crosses a dataflow boundary"
    )
    for ((flow, i) <- io.flows.zipWithIndex) {
      when(writeMatches(i))(assert(flow.bank === writeFlow.bank, "Ambiguous producer storage"))
    }
    writeBank  := writeFlow.bank
    writeRow   := (write.virtualAddress - writeFlow.sourceAddress) >> log2Ceil(bus.bytes)
    writeState := wLocal
  }

  io.ddr.ar.valid                           := readState === rSelect && !readMatches.asUInt.orR
  io.ddr.ar.bits                            := 0.U.asTypeOf(io.ddr.ar.bits)
  io.ddr.ar.bits.addr                       := read.physical
  io.ddr.ar.bits.id                         := read.id
  io.ddr.ar.bits.len                        := read.beats - 1.U
  io.ddr.ar.bits.size                       := log2Ceil(bus.bytes).U
  io.ddr.ar.bits.burst                      := 1.U
  when(io.ddr.ar.fire)(readState            := rExternal)
  io.ddr.aw.valid                           := writeState === wSelect && !writeMatches.asUInt.orR
  io.ddr.aw.bits                            := 0.U.asTypeOf(io.ddr.aw.bits)
  io.ddr.aw.bits.addr                       := write.physical
  io.ddr.aw.bits.id                         := write.id
  io.ddr.aw.bits.len                        := write.beats - 1.U
  io.ddr.aw.bits.size                       := log2Ceil(bus.bytes).U
  io.ddr.aw.bits.burst                      := 1.U
  when(io.ddr.aw.fire)(writeState           := wExternal)
  io.ddr.w.valid                            := writeState === wExternal && writeData.io.output.valid
  io.ddr.w.bits.data                        := writeData.io.output.bits.data
  io.ddr.w.bits.strb                        := writeData.io.output.bits.keep
  io.ddr.w.bits.last                        := writeData.io.output.bits.last
  io.ddr.b.ready                            := writeState === wExternal && io.thread.completed.ready
  io.thread.completed.valid                 := writeState === wComplete || (writeState === wExternal && io.ddr.b.valid)
  io.thread.completed.bits.id               := write.id
  io.thread.completed.bits.error            := Mux(writeState === wExternal, io.ddr.b.bits.resp(1), writeError)
  when(io.thread.completed.fire)(writeState := wIdle)

  val load = io.read.request
  load.valid            := readState === rLocal && reads < read.beats
  load.bits             := 0.U.asTypeOf(load.bits)
  load.bits.bank        := readBank
  load.bits.tuser       := Cat(readRow + reads, false.B)
  load.bits.tag         := reads(1, 0)
  load.bits.tlast       := true.B
  when(load.fire)(reads := reads + 1.U)
  val received = io.read.response
  received.ready                                                        := readState === rLocal && readData.io.input.ready
  io.ddr.r.ready                                                        := readState === rExternal && readData.io.input.ready
  readData.io.input.valid                                               := Mux(readState === rLocal, received.valid, readState === rExternal && io.ddr.r.valid)
  readData.io.input.bits                                                := 0.U.asTypeOf(readData.io.input.bits)
  readData.io.input.bits.id                                             := read.id
  readData.io.input.bits.keep                                           := Fill(bus.bytes, 1.U(1.W))
  readData.io.input.bits.data                                           := Mux(readState === rLocal, received.bits.data, io.ddr.r.bits.data)
  readData.io.input.bits.last                                           := Mux(readState === rLocal, returned +& 1.U === read.beats, io.ddr.r.bits.last)
  readData.io.input.bits.user                                           := Mux(readState === rLocal, Mux(received.bits.error, 2.U, 0.U), io.ddr.r.bits.resp)
  when(received.fire) {
    assert(received.bits.tag === returned(1, 0), "Core memory reordered a stream")
    returned := returned + 1.U
  }
  when(readData.io.input.fire && readData.io.input.bits.last)(readState := rIdle)

  val store = io.write.request
  store.valid               := writeState === wLocal && writes < write.beats && writeData.io.output.valid
  store.bits                := 0.U.asTypeOf(store.bits)
  store.bits.bank           := writeBank
  store.bits.tuser          := Cat(writeRow + writes, true.B)
  store.bits.tag            := writes(1, 0)
  store.bits.data           := writeData.io.output.bits.data
  store.bits.mask           := writeData.io.output.bits.keep
  store.bits.tlast          := true.B
  writeData.io.output.ready := Mux(
    writeState === wLocal,
    writes < write.beats && store.ready,
    writeState === wExternal && io.ddr.w.ready
  )
  when(store.fire) {
    assert(writeData.io.output.bits.last === (writes +& 1.U === write.beats), "Core memory write length mismatch")
    assert(writeData.io.output.bits.keep.andR, "Core dataflow requires complete bank rows")
    writes := writes + 1.U
  }
  io.write.response.ready   := writeState === wLocal
  when(io.write.response.fire) {
    committed                                         := committed + 1.U
    writeError                                        := writeError || io.write.response.bits.error
    when(committed +& 1.U === write.beats)(writeState := wComplete)
  }
  for (bank <- 0 until p.bankCount) {
    io.busy(bank) := (readState === rLocal && readBank === bank.U) ||
      ((writeState === wLocal || writeState === wComplete) && writeBank === bank.U)
  }
}
