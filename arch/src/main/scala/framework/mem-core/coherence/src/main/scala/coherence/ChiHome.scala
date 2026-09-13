package memcore.memory.coherence

import chisel3._
import chisel3.util._
import memcore.bus.chi._

// A finite-address, precise-directory Home. One serialized transaction per Home;
// CompAck closes a read grant before another request can observe its directory state.
// RN nodes are numbered 1..agents. Backing memory is authoritative unless an RN
// holds Unique permission; all competing readers then snoop that RN first.
class ChiHome(
  p:         ChiParams,
  agents:    Int,
  lines:     Int,
  homeId:    Int = 64,
  homeCount: Int = 1,
  homeIndex: Int = 0)
    extends Module {
  require(agents >= 2 && agents < homeId && homeId < (1 << p.nodeIdBits))
  require(lines >= 2 && isPow2(lines))
  require(homeCount <= lines && homeIndex >= 0 && homeIndex < homeCount)
  val mapping    = HomeMapping(homeCount, homeId - homeIndex)
  val localLines = lines / homeCount

  def directoryIndex(addr: UInt): UInt =
    if (localLines == 1) 0.U(0.W)
    else addr(log2Ceil(lines) + 5, mapping.stripeBits + 6)

  val io = IO(new Bundle {
    val req                     = Flipped(Decoupled(new ChiReq(p)))
    val rxRsp                   = Flipped(Decoupled(new ChiRsp(p)))
    val rxDat                   = Flipped(Decoupled(new ChiDat(p)))
    val rsp                     = Decoupled(new ChiRsp(p))
    val dat                     = Decoupled(new ChiDat(p))
    val snp                     = Vec(agents, Decoupled(new ChiSnp(p)))
    val memoryReq               = Decoupled(new LineRequest(p))
    val memoryResp              = Flipped(Decoupled(new LineResponse))
    val busy                    = Output(Bool())
    val cancelledWritebackBeats = Output(UInt(32.W))
  })

  val Seq(
    idle,
    readMem,
    waitRead,
    chooseSnoop,
    sendSnoop,
    waitSnoop,
    writeMem,
    waitWrite,
    grantData,
    waitAck,
    evictResp,
    wbGrant,
    wbData,
    writeResponse
  ) = Enum(14)

  val state               = RegInit(idle)
  val request             = Reg(new ChiReq(p))
  val sharers             = RegInit(VecInit(Seq.fill(localLines)(0.U(agents.W))))
  val lineIndex           = directoryIndex(request.addr)
  val cancelledWritebacks = RegInit(0.U(32.W))
  io.cancelledWritebackBeats := cancelledWritebacks
  val requestMask  = UIntToOH(request.srcId - 1.U, agents)
  val unique       = request.opcode === ChiOpcode.ReadUnique.U
  val maintenance  = request.opcode === ChiOpcode.CleanInvalid.U
  val noSnoopRead  = request.opcode === ChiOpcode.ReadNoSnp.U
  val noSnoopWrite = request.opcode === ChiOpcode.WriteNoSnpFull.U || request.opcode === ChiOpcode.WriteNoSnpPtl.U
  val invalidate   = unique || maintenance
  val pending      = Reg(UInt(agents.W))
  val target       = Reg(UInt(log2Ceil(agents).W))
  val payload      = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val byteMask     = Reg(Vec(p.beatsPerLine, UInt(p.bytesPerBeat.W)))
  val seen         = RegInit(0.U(p.beatsPerLine.W))
  val beat         = RegInit(0.U(math.max(1, log2Ceil(p.beatsPerLine)).W))
  val error        = RegInit(false.B)
  val afterWrite   = Reg(UInt(state.getWidth.W))
  val snoopResp    = Reg(UInt(3.W))
  val writeback    = request.opcode === ChiOpcode.WriteBackFull.U
  io.busy      := state =/= idle
  io.req.ready := state === idle
  when(io.req.fire) {
    val r            = io.req.bits
    assert(r.tgtId === homeId.U && r.srcId >= 1.U && r.srcId <= agents.U, "Invalid coherent requester/Home ID")
    assert(mapping.node(r.addr) === homeId.U, "Request routed to wrong address Home")
    val coherentRead = r.opcode === ChiOpcode.ReadShared.U || r.opcode === ChiOpcode.ReadNotSharedDirty.U ||
      r.opcode === ChiOpcode.ReadUnique.U
    val read         = coherentRead || r.opcode === ChiOpcode.ReadNoSnp.U
    val clean        = r.opcode === ChiOpcode.CleanInvalid.U
    assert(
      read || clean || r.opcode === ChiOpcode.Evict.U || r.opcode === ChiOpcode.WriteBackFull.U ||
        r.opcode === ChiOpcode.WriteNoSnpFull.U || r.opcode === ChiOpcode.WriteNoSnpPtl.U,
      "Unsupported coherent Home opcode"
    )
    assert(
      r.size === 6.U && r.addr(5, 0) === 0.U && r.order === 0.U &&
        r.exclSnoopMe === 0.U && r.pCrdType === 0.U && r.stashNidValidEndian === 0.U,
      "Unsupported coherent Home attributes"
    )
    assert(r.expCompAck === coherentRead.asUInt, "Coherent read grants require CompAck")
    request            := r
    seen               := 0.U
    beat               := 0.U
    error              := r.addr >= (BigInt(lines) * 64).U
    when(read)(state := readMem)
      .elsewhen(clean) {
        pending := Mux(r.addr >= (BigInt(lines) * 64).U, 0.U, sharers(directoryIndex(r.addr)))
        state   := chooseSnoop
      }
      .elsewhen(r.opcode === ChiOpcode.Evict.U)(state := evictResp)
      .otherwise(state := wbGrant)
  }

  io.memoryReq.valid            := state === readMem || state === writeMem
  io.memoryReq.bits             := 0.U.asTypeOf(new LineRequest(p))
  io.memoryReq.bits.addr        := mapping.localAddress(request.addr)
  io.memoryReq.bits.write       := state === writeMem
  io.memoryReq.bits.data        := payload.asUInt
  io.memoryReq.bits.mask        := Mux(noSnoopWrite, byteMask.asUInt, Fill(64, 1.U(1.W)))
  when(io.memoryReq.fire)(state := Mux(state === readMem, waitRead, waitWrite))
  io.memoryResp.ready           := state === waitRead || state === waitWrite
  when(io.memoryResp.fire) {
    assert(io.memoryResp.bits.id === 0.U, "Home received unknown memory response")
    error := error || io.memoryResp.bits.error
    when(state === waitRead) {
      payload := io.memoryResp.bits.data.asTypeOf(payload)
      pending := Mux(error || io.memoryResp.bits.error || noSnoopRead, 0.U, sharers(lineIndex) & ~requestMask)
      state   := chooseSnoop
    }.otherwise {
      // Dirty data accepted from a cache must never be silently discarded.
      when(!noSnoopWrite)(assert(!io.memoryResp.bits.error, "Failed dirty-data writeback at Home"))
      state := afterWrite
    }
  }

  when(state === chooseSnoop) {
    when(pending.orR) {
      target := PriorityEncoder(pending)
      seen   := 0.U
      state  := sendSnoop
    }.otherwise {
      when(!error && !maintenance && !noSnoopRead) {
        sharers(lineIndex) := Mux(unique, requestMask, sharers(lineIndex) | requestMask)
      }
      beat  := 0.U
      state := Mux(maintenance, evictResp, grantData)
    }
  }
  for (i <- 0 until agents) {
    io.snp(i).valid            := state === sendSnoop && target === i.U
    io.snp(i).bits             := 0.U.asTypeOf(new ChiSnp(p))
    io.snp(i).bits.srcId       := homeId.U
    io.snp(i).bits.addr        := request.addr >> 3
    io.snp(i).bits.opcode      := Mux(
      maintenance,
      ChiOpcode.SnpCleanInvalid.U,
      Mux(unique, ChiOpcode.SnpUnique.U, ChiOpcode.SnpNotSharedDirty.U)
    )
    io.snp(i).bits.doNotGoToSd := 1.U
    when(io.snp(i).fire)(state := waitSnoop)
  }

  def finishSnoop(resp: UInt): Unit = {
    assert(
      resp(1, 0) === CoherenceState.I.U || (!invalidate && resp(1, 0) === CoherenceState.SC.U),
      "Snoop failed to revoke/downgrade permission"
    )
    val targetMask = UIntToOH(target, agents)
    pending                                                    := pending & ~targetMask
    when(resp(1, 0) === CoherenceState.I.U)(sharers(lineIndex) := sharers(lineIndex) & ~targetMask)
  }

  io.rxRsp.ready := state === waitSnoop || state === waitAck
  when(io.rxRsp.fire) {
    val r = io.rxRsp.bits
    assert(r.tgtId === homeId.U && r.txnId === 0.U && r.respErr === 0.U, "Invalid Home RSP")
    when(state === waitSnoop) {
      assert(
        r.opcode === ChiOpcode.SnpResp.U && r.srcId === (target +& 1.U) && !r.resp(2),
        "Invalid dataless snoop response"
      )
      finishSnoop(r.resp)
      state := chooseSnoop
    }.otherwise {
      assert(r.opcode === ChiOpcode.CompAck.U && r.srcId === request.srcId, "Invalid CompAck")
      state := idle
    }
  }

  io.rxDat.ready := state === waitSnoop || state === wbData
  when(io.rxDat.fire) {
    val d        = io.rxDat.bits
    val b        = if (p.beatsPerLine == 1) 0.U(0.W) else d.dataId >> log2Ceil(p.dataBits / 128)
    assert(d.tgtId === homeId.U && d.txnId === 0.U && d.respErr === 0.U, "Invalid Home DAT")
    assert(
      VecInit((0 until p.beatsPerLine).map(i => d.dataId === (i * p.dataBits / 128).U)).asUInt.orR,
      "Invalid coherent DataID"
    )
    assert(!(seen & UIntToOH(b, p.beatsPerLine)).orR, "Duplicate coherent DataID")
    val nextSeen = seen | UIntToOH(b, p.beatsPerLine)
    when(seen.orR)(assert(d.resp === snoopResp, "Inconsistent multi-beat cache state"))
    snoopResp   := d.resp
    seen        := nextSeen
    payload(b)  := d.data
    byteMask(b) := d.be
    when(state === waitSnoop) {
      assert(d.opcode === ChiOpcode.SnpRespData.U && d.srcId === (target +& 1.U), "Wrong snoop data source")
      when(nextSeen.andR) {
        finishSnoop(d.resp)
        afterWrite := chooseSnoop
        state      := writeMem
      }
    }.otherwise {
      assert(
        d.opcode === Mux(writeback, ChiOpcode.CopyBackWrData.U, ChiOpcode.NonCopyBackWrData.U) &&
          d.srcId === request.srcId,
        "Wrong write data source/opcode"
      )
      // A snoop may invalidate a victim while WriteBackFull is still queued.
      // CopyBackWrData_I has no valid data and must not overwrite newer memory.
      when(writeback && d.resp === CoherenceState.I.U) {
        assert(d.be === 0.U, "Invalid writeback must not contain valid bytes")
        cancelledWritebacks := cancelledWritebacks + 1.U
      }.elsewhen(request.opcode =/= ChiOpcode.WriteNoSnpPtl.U) {
        assert(d.be.andR, "Full write requires all byte enables")
      }
      when(nextSeen.andR) {
        when(!error && writeback)(sharers(lineIndex) := sharers(lineIndex) & ~requestMask)
        afterWrite                                   := Mux(writeback, idle, writeResponse)
        state                                        := Mux(writeback && d.resp === CoherenceState.I.U, idle, writeMem)
      }
    }
  }

  io.rsp.valid        := state === evictResp || state === wbGrant || state === writeResponse
  io.rsp.bits         := 0.U.asTypeOf(new ChiRsp(p))
  io.rsp.bits.tgtId   := request.srcId
  io.rsp.bits.srcId   := homeId.U
  io.rsp.bits.txnId   := request.txnId
  io.rsp.bits.opcode  := Mux(
    state === wbGrant,
    Mux(writeback, ChiOpcode.CompDBIDResp.U, ChiOpcode.DBIDResp.U),
    ChiOpcode.Comp.U
  )
  io.rsp.bits.respErr := Mux(error, 2.U, 0.U)
  when(io.rsp.fire) {
    when(state === wbGrant) { seen := 0.U; state := wbData }
      .otherwise {
        when(!error && !noSnoopWrite)(sharers(lineIndex) := sharers(lineIndex) & ~requestMask)
        state                                            := idle
      }
  }
  io.dat.valid        := state === grantData
  io.dat.bits         := 0.U.asTypeOf(new ChiDat(p))
  io.dat.bits.tgtId   := request.srcId
  io.dat.bits.srcId   := homeId.U
  io.dat.bits.homeNid := homeId.U
  io.dat.bits.txnId   := request.txnId
  io.dat.bits.opcode  := ChiOpcode.CompData.U
  io.dat.bits.resp    := Mux(error || noSnoopRead, CoherenceState.I.U, Mux(unique, CoherenceState.UC.U, CoherenceState.SC.U))
  io.dat.bits.respErr := Mux(error, 2.U, 0.U)
  io.dat.bits.dataId  := beat * (p.dataBits / 128).U
  val b = if (p.beatsPerLine == 1) 0.U(0.W) else beat
  io.dat.bits.data := Mux(error, 0.U, payload(b))
  when(io.dat.fire) {
    beat                                        := beat + 1.U
    when(beat === (p.beatsPerLine - 1).U)(state := Mux(noSnoopRead, idle, waitAck))
  }
}
