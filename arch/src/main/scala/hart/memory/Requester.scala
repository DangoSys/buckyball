package framework.hart.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.bus.chi._
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.Queue

/** One hart identity. The first cacheClients retain ownership; DMA clients do not answer probes. */
@instantiable
class Requester(
  p:            Params,
  nodeId:       Int,
  localIds:     Seq[Int],
  cacheClients: Int)
    extends Module {
  require(cacheClients > 0 && cacheClients <= localIds.size)
  val ownerBits       = math.max(1, log2Ceil(localIds.size))
  val transactionBits = p.txnIdBits - ownerBits

  @public val io = IO(new Bundle {
    val clients  = Vec(localIds.size, Flipped(new RequesterPort(p)))
    val coherent = new RequesterPort(p)
  })

  val requests      = Instantiate(new RRArbiter(new RequestFlit(p), localIds.size))
  val responses     = Instantiate(new RRArbiter(new ResponseFlit(p), localIds.size + 1))
  val data          = Instantiate(new RRArbiter(new DataFlit(p), localIds.size + 1))
  io.coherent.req <> requests.io.out
  io.coherent.txRsp <> responses.io.out
  io.coherent.txDat <> data.io.out
  val responseOwner = io.coherent.rxRsp.bits.txnId >> transactionBits
  val dataOwner     = io.coherent.rxDat.bits.txnId >> transactionBits
  io.coherent.rxRsp.ready := false.B
  io.coherent.rxDat.ready := false.B

  val idle :: send :: collect :: reply :: returnData :: Nil = Enum(5)
  val phase                                                 = RegInit(idle)
  val probe                                                 = Reg(new SnoopFlit(p))
  val owner                                                 = Reg(UInt(ownerBits.W))
  val result                                                = Reg(UInt(3.W))
  val error                                                 = Reg(UInt(2.W))
  val payload                                               = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val hasData                                               = RegInit(false.B)
  val hasDirtyData                                          = RegInit(false.B)
  val seen                                                  = RegInit(0.U(p.beatsPerLine.W))
  val beat                                                  = RegInit(0.U(log2Ceil(p.beatsPerLine).W))
  io.coherent.snp.ready := phase === idle
  when(io.coherent.snp.fire) {
    probe        := io.coherent.snp.bits
    owner        := 0.U
    result       := 0.U
    error        := 0.U
    hasData      := false.B
    hasDirtyData := false.B
    seen         := 0.U
    phase        := send
  }

  for ((client, index) <- io.clients.zipWithIndex) {
    // Evict is only a local hint. The device directory may conservatively retain
    // this hart until a probe establishes that every local cache has dropped it.
    val evicted  = Instantiate(new Queue(new ResponseFlit(p), 1, false))
    val eviction = client.req.bits.opcode === Opcode.Evict.U
    evicted.io.enq.valid             := client.req.valid && eviction
    evicted.io.enq.bits              := 0.U.asTypeOf(evicted.io.enq.bits)
    evicted.io.enq.bits.srcId        := client.req.bits.tgtId
    evicted.io.enq.bits.tgtId        := localIds(index).U
    evicted.io.enq.bits.txnId        := client.req.bits.txnId
    evicted.io.enq.bits.opcode       := Opcode.Comp.U
    requests.io.in(index).valid      := client.req.valid && !eviction
    requests.io.in(index).bits       := client.req.bits
    requests.io.in(index).bits.srcId := nodeId.U
    requests.io.in(index).bits.txnId := Cat(index.U(ownerBits.W), client.req.bits.txnId(transactionBits - 1, 0))
    when(client.req.valid) {
      assert(client.req.bits.txnId < (BigInt(1) << transactionBits).U, "Local transaction ID exceeds hart namespace")
    }
    when(client.req.bits.opcode === Opcode.WriteBackFull.U) {
      requests.io.in(index).bits.opcode := Opcode.WriteCleanFull.U
    }
    client.req.ready                 := Mux(eviction, evicted.io.enq.ready, requests.io.in(index).ready)
    val returned = Instantiate(new RRArbiter(new ResponseFlit(p), 2))
    returned.io.in(0) <> evicted.io.deq
    returned.io.in(1).valid                                 := io.coherent.rxRsp.valid && responseOwner === index.U
    returned.io.in(1).bits                                  := io.coherent.rxRsp.bits
    returned.io.in(1).bits.tgtId                            := localIds(index).U
    returned.io.in(1).bits.txnId                            := io.coherent.rxRsp.bits.txnId(transactionBits - 1, 0)
    client.rxRsp <> returned.io.out
    when(responseOwner === index.U)(io.coherent.rxRsp.ready := returned.io.in(1).ready)
    client.rxDat.valid                                      := io.coherent.rxDat.valid && dataOwner === index.U
    client.rxDat.bits                                       := io.coherent.rxDat.bits
    client.rxDat.bits.tgtId                                 := localIds(index).U
    client.rxDat.bits.txnId                                 := io.coherent.rxDat.bits.txnId(transactionBits - 1, 0)
    when(dataOwner === index.U)(io.coherent.rxDat.ready     := client.rxDat.ready)

    client.snp.valid            := phase === send && owner === index.U
    client.snp.bits             := probe
    when(client.snp.fire)(phase := collect)
    val snoopResponse = client.txRsp.bits.opcode === Opcode.SnpResp.U
    val snoopData     = client.txDat.bits.opcode === Opcode.SnpRespData.U
    responses.io.in(index).valid      := client.txRsp.valid && !snoopResponse
    responses.io.in(index).bits       := client.txRsp.bits
    responses.io.in(index).bits.srcId := nodeId.U
    client.txRsp.ready                := Mux(snoopResponse, phase === collect && owner === index.U, responses.io.in(index).ready)
    data.io.in(index).valid           := client.txDat.valid && !snoopData
    data.io.in(index).bits            := client.txDat.bits
    data.io.in(index).bits.srcId      := nodeId.U
    client.txDat.ready                := Mux(snoopData, phase === collect && owner === index.U, data.io.in(index).ready)
    val receivedResponse = client.txRsp.fire && snoopResponse
    val receivedData     = client.txDat.fire && snoopData
    val dataBeat         = client.txDat.bits.dataId >> log2Ceil(p.dataBits / 128)
    val received         = seen | UIntToOH(dataBeat, p.beatsPerLine)
    when(receivedResponse) {
      assert(client.txRsp.bits.txnId === probe.txnId, "Hart probe response transaction mismatch")
      result := result | client.txRsp.bits.resp
      error  := error | client.txRsp.bits.respErr
    }
    when(receivedData) {
      assert(client.txDat.bits.txnId === probe.txnId && client.txDat.bits.be.andR, "Invalid hart probe data")
      assert(!seen(dataBeat), "Duplicate hart probe beat")
      when(!hasDirtyData || client.txDat.bits.resp(2))(payload(dataBeat) := client.txDat.bits.data)
      seen                                                               := received
      result                                                             := result | client.txDat.bits.resp
      error                                                              := error | client.txDat.bits.respErr
      hasData                                                            := true.B
      when(received.andR)(hasDirtyData                                   := hasDirtyData || client.txDat.bits.resp(2))
    }
    when(receivedResponse || receivedData && received.andR) {
      seen := 0.U
      when(owner === (cacheClients - 1).U) {
        beat  := 0.U
        phase := Mux(hasData || receivedData, returnData, reply)
      }.otherwise {
        owner := owner + 1.U
        phase := send
      }
    }
  }
  responses.io.in(localIds.size).valid := phase === reply
  responses.io.in(localIds.size).bits             := 0.U.asTypeOf(responses.io.in(localIds.size).bits)
  responses.io.in(localIds.size).bits.srcId       := nodeId.U
  responses.io.in(localIds.size).bits.tgtId       := probe.srcId
  responses.io.in(localIds.size).bits.txnId       := probe.txnId
  responses.io.in(localIds.size).bits.opcode      := Opcode.SnpResp.U
  responses.io.in(localIds.size).bits.resp        := result
  responses.io.in(localIds.size).bits.respErr     := error
  when(responses.io.in(localIds.size).fire)(phase := idle)
  data.io.in(localIds.size).valid                 := phase === returnData
  data.io.in(localIds.size).bits                  := 0.U.asTypeOf(data.io.in(localIds.size).bits)
  data.io.in(localIds.size).bits.srcId            := nodeId.U
  data.io.in(localIds.size).bits.tgtId            := probe.srcId
  data.io.in(localIds.size).bits.txnId            := probe.txnId
  data.io.in(localIds.size).bits.opcode           := Opcode.SnpRespData.U
  data.io.in(localIds.size).bits.resp             := result
  data.io.in(localIds.size).bits.respErr          := error
  data.io.in(localIds.size).bits.data             := payload(beat)
  data.io.in(localIds.size).bits.dataId           := beat << log2Ceil(p.dataBits / 128)
  data.io.in(localIds.size).bits.be               := Fill(p.bytesPerBeat, 1.U(1.W))
  when(data.io.in(localIds.size).fire) {
    when(beat === (p.beatsPerLine - 1).U)(phase := idle).otherwise(beat := beat + 1.U)
  }
}
