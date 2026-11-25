package examples.hart.magia

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.bus.{axi4, chi}
import framework.generic.bus.chi.{HomeMapping, Opcode}
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.Queue

/**
 * The mesh's AXI master on the coherent bus: AXI bursts become non-allocating coherent transactions, with one
 * in-flight line per direction, so the mesh reads what the Loom wrote through its caches.
 */
@instantiable
class DmaRequester(
  bus:     axi4.Params,
  p:       chi.Params,
  nodeId:  Int,
  mapping: HomeMapping)
    extends Module {
  require(bus.dataBits == 128 && p.dataBits == 256 && bus.addressBits == p.addressBits)

  @public val io = IO(new Bundle {
    val axi      = Flipped(new axi4.Port(bus))
    val coherent = new chi.RequesterPort(p)
  })

  val readIdle :: readRequest :: readWait :: readReturn :: Nil                                               = Enum(4)
  val writeIdle :: writeCollect :: writeRequest :: writeDbid :: writeSend :: writeWait :: writeReturn :: Nil = Enum(7)
  val readState                                                                                              = RegInit(readIdle)
  val writeState                                                                                             = RegInit(writeIdle)
  val read                                                                                                   = Reg(chiselTypeOf(io.axi.ar.bits))
  val write                                                                                                  = Reg(chiselTypeOf(io.axi.aw.bits))
  val readLeft                                                                                               = Reg(UInt(8.W))
  val writeLeft                                                                                              = Reg(UInt(8.W))
  val readLine                                                                                               = Reg(Vec(2, UInt(256.W)))
  val readSeen                                                                                               = RegInit(0.U(2.W))
  val readError                                                                                              = RegInit(false.B)
  val writeLine                                                                                              = Reg(Vec(4, UInt(128.W)))
  val writeMask                                                                                              = RegInit(VecInit(Seq.fill(4)(0.U(16.W))))
  val writeAddress                                                                                           = Reg(UInt(bus.addressBits.W))
  val writeLast                                                                                              = Reg(Bool())
  val writeError                                                                                             = RegInit(false.B)
  val dbid                                                                                                   = Reg(UInt(p.dbIdBits.W))
  val writeBeat                                                                                              = RegInit(false.B)

  io.axi.ar.ready := readState === readIdle
  when(io.axi.ar.fire) {
    assert(
      io.axi.ar.bits.burst === 1.U && io.axi.ar.bits.size <= 4.U && !io.axi.ar.bits.lock,
      "Unsupported DMA read burst"
    )
    read      := io.axi.ar.bits
    readLeft  := io.axi.ar.bits.len
    readSeen  := 0.U
    readError := false.B
    readState := readRequest
  }
  io.axi.aw.ready := writeState === writeIdle
  when(io.axi.aw.fire) {
    assert(
      io.axi.aw.bits.burst === 1.U && io.axi.aw.bits.size <= 4.U && !io.axi.aw.bits.lock,
      "Unsupported DMA write burst"
    )
    write               := io.axi.aw.bits
    writeLeft           := io.axi.aw.bits.len
    writeAddress        := io.axi.aw.bits.addr
    writeMask.foreach(_ := 0.U)
    writeError          := false.B
    writeState          := writeCollect
  }
  io.axi.w.ready  := writeState === writeCollect
  when(io.axi.w.fire) {
    assert(io.axi.w.bits.last === (writeLeft === 0.U), "DMA write burst length mismatch")
    val slot = write.addr(5, 4)
    writeLine(slot) := VecInit((0 until 16).map(byte =>
      Mux(io.axi.w.bits.strb(byte), io.axi.w.bits.data(8 * byte + 7, 8 * byte), writeLine(slot)(8 * byte + 7, 8 * byte))
    )).asUInt
    writeMask(slot) := writeMask(slot) | io.axi.w.bits.strb
    val next = write.addr + (1.U << write.size)
    write.addr := next
    writeLeft  := writeLeft - 1.U
    when(writeLeft === 0.U || next(5, 0) === 0.U) {
      writeLast  := writeLeft === 0.U
      writeState := writeRequest
    }
  }
  val requests = Instantiate(new RRArbiter(new chi.RequestFlit(p), 2))
  io.coherent.req <> requests.io.out
  for (i <- 0 until 2) {
    val request = requests.io.in(i)
    request.valid := (if (i == 0) readState === readRequest else writeState === writeRequest)
    request.bits  := 0.U.asTypeOf(request.bits)
    val address = if (i == 0) read.addr else writeAddress
    request.bits.srcId      := nodeId.U
    request.bits.tgtId      := mapping.node(address)
    request.bits.txnId      := i.U
    request.bits.addr       := (address >> 6) << 6
    request.bits.size       := 6.U
    request.bits.opcode     := (if (i == 0) Opcode.ReadOnce else Opcode.WriteUniquePtl).U
    request.bits.snpAttr    := 1.U
    request.bits.memAttr    := "b1100".U
    request.bits.allowRetry := 1.U
  }
  when(requests.io.in(0).fire)(readState := readWait)
  when(requests.io.in(1).fire)(writeState := writeDbid)
  io.coherent.rxDat.ready                 := readState === readWait
  when(io.coherent.rxDat.fire) {
    val value    = io.coherent.rxDat.bits
    assert(value.txnId === 0.U && value.opcode === Opcode.CompData.U, "Unexpected DMA read response")
    val slot     = value.dataId(1)
    val received = readSeen | UIntToOH(slot, 2)
    assert(!readSeen(slot), "Duplicate DMA read response beat")
    readLine(slot)                := value.data
    readSeen                      := received
    readError                     := readError || value.respErr =/= 0.U
    when(received.andR)(readState := readReturn)
  }
  io.axi.r.valid                          := readState === readReturn
  io.axi.r.bits                           := 0.U.asTypeOf(io.axi.r.bits)
  io.axi.r.bits.id                        := read.id
  io.axi.r.bits.data                      := (readLine.asUInt >> (read.addr(5, 4) << 7))(127, 0)
  io.axi.r.bits.resp                      := Mux(readError, 2.U, 0.U)
  io.axi.r.bits.last                      := readLeft === 0.U
  when(io.axi.r.fire) {
    val next = read.addr + (1.U << read.size)
    read.addr := next
    readLeft  := readLeft - 1.U
    when(readLeft === 0.U)(readState := readIdle)
      .elsewhen(next(5, 0) === 0.U) {
        readSeen  := 0.U
        readError := false.B
        readState := readRequest
      }
  }
  io.coherent.rxRsp.ready                 := writeState === writeDbid || writeState === writeWait
  when(io.coherent.rxRsp.fire) {
    val value = io.coherent.rxRsp.bits
    assert(value.txnId === 1.U, "Unexpected DMA write transaction")
    writeError := writeError || value.respErr =/= 0.U
    when(writeState === writeDbid) {
      assert(value.opcode === Opcode.DBIDResp.U, "DMA write requires DBIDResp")
      dbid       := value.dbid
      writeBeat  := false.B
      writeState := writeSend
    }.otherwise {
      assert(value.opcode === Opcode.Comp.U, "DMA write requires completion")
      when(writeLast)(writeState := writeReturn).otherwise {
        writeAddress        := write.addr
        writeMask.foreach(_ := 0.U)
        writeState          := writeCollect
      }
    }
  }
  io.coherent.txDat.valid                 := writeState === writeSend
  io.coherent.txDat.bits                  := 0.U.asTypeOf(io.coherent.txDat.bits)
  io.coherent.txDat.bits.srcId            := nodeId.U
  io.coherent.txDat.bits.tgtId            := mapping.node(writeAddress)
  io.coherent.txDat.bits.txnId            := dbid
  io.coherent.txDat.bits.opcode           := Opcode.NonCopyBackWriteData.U
  io.coherent.txDat.bits.dataId           := Mux(writeBeat, 2.U, 0.U)
  io.coherent.txDat.bits.data             := Mux(writeBeat, Cat(writeLine(3), writeLine(2)), Cat(writeLine(1), writeLine(0)))
  io.coherent.txDat.bits.be               := Mux(writeBeat, Cat(writeMask(3), writeMask(2)), Cat(writeMask(1), writeMask(0)))
  when(io.coherent.txDat.fire) {
    writeBeat                  := true.B
    when(writeBeat)(writeState := writeWait)
  }
  io.axi.b.valid                          := writeState === writeReturn
  io.axi.b.bits                           := 0.U.asTypeOf(io.axi.b.bits)
  io.axi.b.bits.id                        := write.id
  io.axi.b.bits.resp                      := Mux(writeError, 2.U, 0.U)
  when(io.axi.b.fire)(writeState          := writeIdle)

  // No line remains resident after a transaction; this client never owns a directory entry.
  val probes = Instantiate(new Queue(new chi.ResponseFlit(p), 1, false))
  probes.io.enq.valid       := io.coherent.snp.valid
  io.coherent.snp.ready     := probes.io.enq.ready
  probes.io.enq.bits        := 0.U.asTypeOf(probes.io.enq.bits)
  probes.io.enq.bits.srcId  := nodeId.U
  probes.io.enq.bits.tgtId  := io.coherent.snp.bits.srcId
  probes.io.enq.bits.txnId  := io.coherent.snp.bits.txnId
  probes.io.enq.bits.opcode := Opcode.SnpResp.U
  io.coherent.txRsp <> probes.io.deq
}
