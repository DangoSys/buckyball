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
    val thread   = Flipped(new Dma(bus))
    val task     = Input(UInt(64.W))
    val flows    = Input(Vec(p.bankCount, new Flow(p)))
    val read     = Flipped(new MeshChannel(p))
    val write    = Flipped(new MeshChannel(p))
    val external = new Dma(bus)
    val busy     = Output(Vec(p.bankCount, Bool()))
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
  val externalRead                                                = Instantiate(new Receive(new axi.Beat(bus.dataBits, bus.idBits, 0, 2)))
  val externalWrite                                               = Instantiate(new Send(new axi.Beat(bus.dataBits)))
  externalRead.io.input <> io.external.readData
  io.external.writeData <> externalWrite.io.output
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

  io.external.read.valid                    := readState === rSelect && !readMatches.asUInt.orR
  io.external.read.bits                     := read
  when(io.external.read.fire)(readState     := rExternal)
  io.external.write.valid                   := writeState === wSelect && !writeMatches.asUInt.orR
  io.external.write.bits                    := write
  when(io.external.write.fire)(writeState   := wExternal)
  externalWrite.io.input.valid              := writeState === wExternal && writeData.io.output.valid
  externalWrite.io.input.bits               := writeData.io.output.bits
  io.external.completed.ready               := writeState === wExternal && io.thread.completed.ready
  io.thread.completed.valid                 := writeState === wComplete || (writeState === wExternal && io.external.completed.valid)
  io.thread.completed.bits.id               := write.id
  io.thread.completed.bits.error            := Mux(writeState === wExternal, io.external.completed.bits.error, writeError)
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
  externalRead.io.output.ready                                          := readState === rExternal && readData.io.input.ready
  readData.io.input.valid                                               := Mux(
    readState === rLocal,
    received.valid,
    readState === rExternal && externalRead.io.output.valid
  )
  readData.io.input.bits                                                := 0.U.asTypeOf(readData.io.input.bits)
  readData.io.input.bits.id                                             := read.id
  readData.io.input.bits.keep                                           := Fill(bus.bytes, 1.U(1.W))
  readData.io.input.bits.data                                           := Mux(readState === rLocal, received.bits.data, externalRead.io.output.bits.data)
  readData.io.input.bits.last                                           := Mux(
    readState === rLocal,
    returned +& 1.U === read.beats,
    externalRead.io.output.bits.last
  )
  readData.io.input.bits.user                                           := Mux(
    readState === rLocal,
    Mux(received.bits.error, 2.U, 0.U),
    externalRead.io.output.bits.user
  )
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
    writeState === wExternal && externalWrite.io.input.ready
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
