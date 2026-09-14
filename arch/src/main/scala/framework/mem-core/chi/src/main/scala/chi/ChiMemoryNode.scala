package memcore.bus.chi

import chisel3._
import chisel3.util._

class LineRequest(p: ChiParams) extends Bundle {
  val id    = UInt(8.W)
  val addr  = UInt(p.addressBits.W)
  val write = Bool()
  val data  = UInt(512.W)
  val mask  = UInt(64.W)
}

class LineResponse extends Bundle {
  val id    = UInt(8.W)
  val data  = UInt(512.W)
  val error = Bool()
}

// SN-side transaction subset: aligned 64-byte ReadNoSnp and WriteNoSnpFull/Ptl,
// Order=00, ExpCompAck=0, no exclusive/atomic/retry/cancellation transactions.
// The Home must perform any coherence work before forwarding a request here.
class ChiMemoryNode(p: ChiParams, nodeId: Int = 1, slots: Int = 8) extends Module {
  require(nodeId >= 0 && BigInt(nodeId) < (BigInt(1) << p.nodeIdBits))
  require(slots >= 1 && slots <= 256)

  val io = IO(new Bundle {
    val req         = Flipped(Decoupled(new ChiReq(p)))
    val rxDat       = Flipped(Decoupled(new ChiDat(p)))
    val rsp         = Decoupled(new ChiRsp(p))
    val txDat       = Decoupled(new ChiDat(p))
    val memoryReq   = Decoupled(new LineRequest(p))
    val memoryResp  = Flipped(Decoupled(new LineResponse))
    val outstanding = Output(UInt(log2Ceil(slots + 1).W))
  })

  val free :: dbid :: collect :: issue :: waitMemory :: complete :: returnData :: Nil = Enum(7)
  val state                                                                           = RegInit(VecInit(Seq.fill(slots)(free)))
  val requests                                                                        = Reg(Vec(slots, new ChiReq(p)))
  val payload                                                                         = Reg(Vec(slots, Vec(p.beatsPerLine, UInt(p.dataBits.W))))
  val masks                                                                           = Reg(Vec(slots, Vec(p.beatsPerLine, UInt(p.bytesPerBeat.W))))
  val received                                                                        = RegInit(VecInit(Seq.fill(slots)(0.U(p.beatsPerLine.W))))
  val sent                                                                            = RegInit(VecInit(Seq.fill(slots)(0.U(3.W))))
  val errors                                                                          = RegInit(VecInit(Seq.fill(slots)(false.B)))
  // Range is checked on the full protocol ID before indexing a local slot.
  def slotIndex(id: UInt): UInt = if (slots == 1) 0.U(0.W) else id(log2Ceil(slots) - 1, 0)
  val freeMask  = VecInit(state.map(_ === free))
  val allocated = PriorityEncoder(freeMask)
  io.req.ready   := freeMask.asUInt.orR
  io.outstanding := PopCount(state.map(_ =/= free))
  when(io.req.fire) {
    val r = io.req.bits
    assert(r.tgtId === nodeId.U, "CHI request to wrong node")
    assert(
      r.opcode === ChiOpcode.ReadNoSnp.U || r.opcode === ChiOpcode.WriteNoSnpFull.U ||
        r.opcode === ChiOpcode.WriteNoSnpPtl.U,
      "Unsupported CHI request opcode"
    )
    assert(r.size === 6.U && r.addr(5, 0) === 0.U, "CHI node requires aligned 64-byte access")
    assert(
      r.order === 0.U && r.expCompAck === 0.U && r.exclSnoopMe === 0.U &&
        r.snpAttr === 0.U && r.stashNidValidEndian === 0.U && r.pCrdType === 0.U,
      "Unsupported CHI request attributes"
    )
    // Normal, non-Device memory only. Allocate/Cacheable/EWA are otherwise accepted.
    assert(!r.memAttr(1), "Device memory is outside this CHI node profile")
    for (i <- 0 until slots) {
      assert(
        !(state(i) =/= free && requests(i).srcId === r.srcId && requests(i).txnId === r.txnId),
        "CHI source reused a live TxnID"
      )
    }
    requests(allocated) := r
    state(allocated)    := Mux(r.opcode === ChiOpcode.ReadNoSnp.U, issue, dbid)
    received(allocated) := 0.U
    sent(allocated)     := 0.U
    errors(allocated)   := false.B
  }

  // DBID is the allocated transaction slot, NOT the original request TxnID.
  // Accept interleaved data from different transactions and out-of-order DataIDs.
  io.rxDat.ready := true.B
  when(io.rxDat.fire) {
    val d       = io.rxDat.bits
    assert(d.opcode === ChiOpcode.NonCopyBackWrData.U, "Unsupported CHI data opcode")
    assert(d.tgtId === nodeId.U && d.txnId < slots.U, "Invalid CHI write destination/DBID")
    assert(d.respErr === 0.U, "Errored write data is outside this CHI node profile")
    val index   = slotIndex(d.txnId)
    val beat    = if (p.beatsPerLine == 1) 0.U(0.W) else d.dataId >> log2Ceil(p.dataBits / 128)
    val legalId = VecInit((0 until p.beatsPerLine).map(i => d.dataId === (i * p.dataBits / 128).U)).asUInt.orR
    assert(legalId, "Invalid CHI DataID for data width")
    when(d.txnId < slots.U) {
      assert(state(index) === collect, "CHI write data before DBID or after completion")
      assert(d.srcId === requests(index).srcId, "CHI write data from wrong requester")
      assert(!(received(index) & UIntToOH(beat, p.beatsPerLine)).orR, "Duplicate CHI DataID")
      when(requests(index).opcode === ChiOpcode.WriteNoSnpFull.U) {
        assert(d.be.andR, "WriteNoSnpFull requires all byte enables")
      }
      payload(index)(beat) := d.data
      masks(index)(beat)   := d.be
      val nextReceived = received(index) | UIntToOH(beat, p.beatsPerLine)
      received(index)                      := nextReceived
      when(nextReceived.andR)(state(index) := issue)
    }
  }

  val memArb = Module(new RRArbiter(new LineRequest(p), slots))
  for (i <- 0 until slots) {
    memArb.io.in(i).valid               := state(i) === issue
    memArb.io.in(i).bits.id             := i.U
    memArb.io.in(i).bits.addr           := requests(i).addr
    memArb.io.in(i).bits.write          := requests(i).opcode =/= ChiOpcode.ReadNoSnp.U
    memArb.io.in(i).bits.data           := payload(i).asUInt
    memArb.io.in(i).bits.mask           := masks(i).asUInt
    when(memArb.io.in(i).fire)(state(i) := waitMemory)
  }
  io.memoryReq <> memArb.io.out
  io.memoryResp.ready := true.B
  when(io.memoryResp.fire) {
    val r     = io.memoryResp.bits
    val index = slotIndex(r.id)
    assert(r.id < slots.U, "Invalid memory response ID")
    when(r.id < slots.U) {
      assert(state(index) === waitMemory, "Unexpected memory completion")
      payload(index) := r.data.asTypeOf(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
      errors(index)  := r.error
      state(index)   := Mux(requests(index).opcode === ChiOpcode.ReadNoSnp.U, returnData, complete)
    }
  }

  val rspArb = Module(new RRArbiter(new ChiRsp(p), slots))
  for (i <- 0 until slots) {
    val r = rspArb.io.in(i)
    r.valid               := state(i) === dbid || state(i) === complete
    r.bits                := 0.U.asTypeOf(new ChiRsp(p))
    r.bits.qos            := requests(i).qos
    r.bits.tgtId          := requests(i).srcId
    r.bits.srcId          := nodeId.U
    r.bits.txnId          := requests(i).txnId
    r.bits.opcode         := Mux(state(i) === dbid, ChiOpcode.DBIDResp.U, ChiOpcode.Comp.U)
    r.bits.dbid           := i.U
    r.bits.respErr        := Mux(errors(i), 2.U, 0.U) // NDERR
    r.bits.traceTag       := requests(i).traceTag
    when(r.fire)(state(i) := Mux(state(i) === dbid, collect, free))
  }
  io.rsp <> rspArb.io.out

  val datArb = Module(new RRArbiter(new ChiDat(p), slots))
  for (i <- 0 until slots) {
    val d = datArb.io.in(i)
    d.valid        := state(i) === returnData
    d.bits         := 0.U.asTypeOf(new ChiDat(p))
    d.bits.qos     := requests(i).qos
    d.bits.srcId   := nodeId.U
    // DMT routing follows the forwarded request's ReturnNID/ReturnTxnID.
    d.bits.tgtId   := requests(i).returnNid
    d.bits.txnId   := requests(i).returnTxnId
    d.bits.homeNid := requests(i).srcId
    d.bits.dbid    := requests(i).txnId
    d.bits.opcode  := ChiOpcode.CompData.U
    d.bits.dataId  := sent(i) * (p.dataBits / 128).U
    d.bits.respErr := Mux(errors(i), 2.U, 0.U)
    val beatIndex = if (p.beatsPerLine == 1) 0.U(0.W) else sent(i)(log2Ceil(p.beatsPerLine) - 1, 0)
    d.bits.data     := Mux(errors(i), 0.U, payload(i)(beatIndex))
    d.bits.traceTag := requests(i).traceTag
    when(d.fire) {
      sent(i)                                           := sent(i) + 1.U
      when(sent(i) === (p.beatsPerLine - 1).U)(state(i) := free)
    }
  }
  io.txDat <> datArb.io.out
}

// Synthesizable 64-byte-line SRAM backend. Each request has a completion,
// including writes. Out-of-range accesses return an error, never wrap.
class ChiLineSram(p: ChiParams, lines: Int = 256) extends Module {
  require(lines >= 2 && isPow2(lines))

  val io = IO(new Bundle {
    val req  = Flipped(Decoupled(new LineRequest(p)))
    val resp = Decoupled(new LineResponse)
  })

  val mem                               = SyncReadMem(lines, Vec(64, UInt(8.W)))
  val idle :: capture :: respond :: Nil = Enum(3)
  val state                             = RegInit(idle)
  val request                           = Reg(new LineRequest(p))
  val result                            = Reg(UInt(512.W))
  val error                             = Reg(Bool())
  io.req.ready := state === idle
  val validAddress = io.req.bits.addr < (BigInt(lines) * 64).U && io.req.bits.addr(5, 0) === 0.U
  val read         = mem.read(io.req.bits.addr(log2Ceil(lines) + 5, 6), io.req.fire && !io.req.bits.write && validAddress)
  when(io.req.fire) {
    request := io.req.bits
    error   := !validAddress
    state   := capture
    when(io.req.bits.write && validAddress) {
      mem.write(
        io.req.bits.addr(log2Ceil(lines) + 5, 6),
        io.req.bits.data.asTypeOf(Vec(64, UInt(8.W))),
        io.req.bits.mask.asBools
      )
    }
  }
  when(state === capture) {
    result := Mux(request.write || error, 0.U, read.asUInt)
    state  := respond
  }
  io.resp.valid := state === respond
  io.resp.bits.id          := request.id
  io.resp.bits.data        := result
  io.resp.bits.error       := error
  when(io.resp.fire)(state := idle)
}
