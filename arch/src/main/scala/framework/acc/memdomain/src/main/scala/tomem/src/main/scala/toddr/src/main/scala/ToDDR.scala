package framework.acc.memdomain.toddr

import framework.core.memory.access.preflight.configs.PmpParams

import framework.acc.memdomain.tomem.configs.{Params => ModuleParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.memory.access.preflight.{MapQuery, MapResult}
import framework.ip.bus.axi4

@instantiable
class ToDDR(val b: ModuleParam, permission: PmpParams) extends Module {
  val p = b

  val prepared  = p.mapping
  val axiParams = p.axi

  @public
  val io = IO(new Bundle {
    val dma =
      Flipped(new DmaPort(b, permission))

    // One decision for each physical producer: loader, storer, kernel image.
    val decisions = Input(Vec(
      3,
      new DmaDecision(b)
    ))

    val queries = Output(Vec(
      2,
      new MapQuery(b.mapping)
    ))

    val mappings = Input(Vec(
      2,
      new MapResult(b.mapping)
    ))

    val axi = new axi4.Port(axiParams)

  })

  val reader: Instance[ReadDma]  = Instantiate(new ReadDma(b))
  val writer: Instance[WriteDma] = Instantiate(new WriteDma(b, permission))
  val kernel       = io.dma.read.bits.producer === 2.U
  val readDecision = Mux(kernel, io.decisions(2), io.decisions(0))
  when(io.dma.read.valid) {
    assert(io.dma.read.bits.producer === 0.U || kernel, "DMA read has an unknown producer")
  }
  reader.io.req.valid     := io.dma.read.valid
  reader.io.req.bits      := io.dma.read.bits.transfer
  io.dma.read.ready       := reader.io.req.ready
  io.dma.readResult <> reader.io.resp
  reader.io.parentId      := readDecision.parentId
  reader.io.decisionValid := readDecision.valid
  reader.io.decisionFault := readDecision.fault
  writer.io.req <> io.dma.write
  writer.io.data <> io.dma.writeData
  io.dma.writeResult <> writer.io.resp
  writer.io.parentId      := io.decisions(1).parentId
  writer.io.decisionValid := io.decisions(1).valid
  writer.io.decisionFault := io.decisions(1).fault
  io.dma.readBusy         := reader.io.busy
  io.dma.writeBusy        := writer.io.busy
  io.queries(0)           := reader.io.query
  io.queries(1)           := writer.io.query
  reader.io.mapping       := io.mappings(0)
  writer.io.mapping       := io.mappings(1)

  // AXI read and write channels retain independent command owners.
  io.axi.ar <> reader.io.axi.ar
  reader.io.axi.r <> io.axi.r
  io.axi.aw <> writer.io.axi.aw
  io.axi.w <> writer.io.axi.w
  writer.io.axi.b <> io.axi.b
  reader.io.axi.aw.ready := false.B
  reader.io.axi.w.ready  := false.B
  reader.io.axi.b.valid  := false.B
  reader.io.axi.b.bits   := 0.U.asTypeOf(new axi4.WriteResponse(axiParams))
  writer.io.axi.ar.ready := false.B
  writer.io.axi.r.valid  := false.B
  writer.io.axi.r.bits   := 0.U.asTypeOf(new axi4.ReadData(axiParams))
}
