package memcore.memory.coherence

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.bus.chi._
import memcore.bus.chi.snf.LineRequest
import memcore.memory.cache.{Cache, CacheOp, CacheRequest}
import memcore.memory.coherence.configs.CoherenceParams

@instantiable
class Coherence(p: CoherenceParams) extends Module {
  @public
  val io = IO(new CoherenceIO(p))
  val cache: Instance[Cache] = Instantiate(new Cache(p.cache))
  val mshr:  Instance[Mshr]  = Instantiate(new Mshr(p))
  val c = p.chi
  val n = p.mshrEntries

  private val states = Enum(23)

  val Seq(
    lookupReq,
    lookupWait,
    chooseSnoop,
    sendSnoop,
    waitSnoop,
    writeMemory,
    waitWrite,
    invalidateReq,
    invalidateWait,
    readMemory,
    waitRead,
    fillReq,
    fillWait,
    updateReq,
    updateWait,
    grantData,
    waitAck,
    complete,
    wbGrant,
    wbData,
    finished
  ) = states.take(21)

  val Seq(restoreReq, restoreWait) = states.drop(21)

  val state         = RegInit(VecInit(Seq.fill(n)(lookupReq)))
  val way           = Reg(Vec(n, UInt(p.cache.wayBits.W)))
  val resident      = Reg(Vec(n, Bool()))
  val victim        = Reg(Vec(n, Bool()))
  val workAddress   = Reg(Vec(n, UInt(c.addressBits.W)))
  val dirty         = Reg(Vec(n, Bool()))
  val payload       = Reg(Vec(n, Vec(c.beatsPerLine, UInt(c.dataBits.W))))
  val pending       = Reg(Vec(n, UInt(p.agents.W)))
  val target        = Reg(Vec(n, UInt(math.max(1, log2Ceil(p.agents)).W)))
  val seen          = Reg(Vec(n, UInt(c.beatsPerLine.W)))
  val sent          = Reg(Vec(n, UInt(math.max(1, log2Ceil(c.beatsPerLine)).W)))
  val snoopResponse = Reg(Vec(n, UInt(3.W)))
  val failed        = Reg(Vec(n, Bool()))
  val directory: Instance[Directory] = Instantiate(new Directory(p))
  val entries   = mshr.io.entries
  val completed = VecInit((0 until n).map(i => entries(i).valid && state(i) === finished)).asUInt
  mshr.io.release.valid         := completed.orR
  mshr.io.release.bits          := (if (n == 1) 0.U else PriorityEncoder(completed))
  mshr.io.allocate.valid        := io.req.valid
  mshr.io.allocate.bits.request := io.req.bits
  mshr.io.allocate.bits.key     := io.req.bits.addr(p.cache.offsetBits + p.cache.setBits - 1, p.cache.offsetBits)
  io.req.ready                  := mshr.io.allocate.ready
  io.outstanding                := mshr.io.outstanding
  when(io.req.fire) {
    val r    = io.req.bits
    val read =
      r.opcode === Opcode.ReadShared.U || r.opcode === Opcode.ReadNotSharedDirty.U || r.opcode === Opcode.ReadUnique.U
    assert(r.tgtId === p.homeId.U && r.srcId >= 1.U && r.srcId <= p.agents.U, "Unknown CPU or Home")
    assert(
      read || r.opcode === Opcode.Evict.U || r.opcode === Opcode.WriteBackFull.U ||
        r.opcode === Opcode.CleanShared.U || r.opcode === Opcode.CleanInvalid.U || r.opcode === Opcode.MakeInvalid.U,
      "Unsupported coherence operation"
    )
    assert(
      r.size === 6.U && r.addr(5, 0) === 0.U && r.order === 0.U && r.pas === 0.U &&
        r.returnNid === 0.U && r.returnTxnId === 0.U && r.memAttr === "b1100".U && r.snpAttr === 1.U &&
        r.allowRetry === 1.U && r.exclSnoopMe === 0.U && r.pCrdType === 0.U && r.stashNidValidEndian === 0.U && r.multiReq === 0.U && r.tagOp === 0.U,
      "Unsupported coherent access attributes"
    )
    assert(r.expCompAck === read.asUInt, "Coherent read requires CompAck")
    val i    = mshr.io.allocatedId
    state(i)  := lookupReq
    failed(i) := false.B
    sent(i)   := 0.U
    seen(i)   := 0.U
  }

  private def arbitrate[T <: Data](gen: T): RRArbiter[T] = Module(new RRArbiter(gen, n) {

    override lazy val lastGrant = {
      val pointer = RegInit(0.U(math.max(1, log2Ceil(this.n)).W))
      when(this.io.out.fire)(pointer := this.io.chosen)
      pointer
    }

  })

  val cacheArb   = arbitrate(new CacheRequest(p.cache))
  val cacheQueue = Module(new Queue(new CacheRequest(p.cache), 2, pipe = true))
  cacheQueue.io.enq <> cacheArb.io.out
  cache.io.request <> cacheQueue.io.deq
  cache.io.response.ready := true.B
  val memArb   = arbitrate(new LineRequest(c))
  val memQueue = Module(new Queue(new LineRequest(c), 2, pipe = true))
  memQueue.io.enq <> memArb.io.out
  io.memoryReq <> memQueue.io.deq
  io.memoryResp.ready := true.B
  val snpArb   = arbitrate(new DirectedSnoop(c))
  val snpQueue = Module(new Queue(new DirectedSnoop(c), 2, pipe = true))
  snpQueue.io.enq <> snpArb.io.out
  io.snp <> snpQueue.io.deq
  val rspArb   = arbitrate(new ResponseFlit(c))
  val rspQueue = Module(new Queue(new ResponseFlit(c), 2, pipe = true))
  rspQueue.io.enq <> rspArb.io.out
  io.rsp <> rspQueue.io.deq
  val datArb   = arbitrate(new DataFlit(c))
  val datQueue = Module(new Queue(new DataFlit(c), 2, pipe = true))
  datQueue.io.enq <> datArb.io.out
  io.dat <> datQueue.io.deq
  // Caller TxnID ends at its final response. The slot/DBID/set stays owned until full protocol completion.
  mshr.io.releaseCaller := VecInit((0 until n).map { i =>
    (io.rsp.fire && io.rsp.bits.opcode === Opcode.CompDBIDResp.U && io.rsp.bits.dbid === i.U) ||
    (io.dat.fire && io.dat.bits.opcode === Opcode.CompData.U && io.dat.bits.dbid === i.U &&
      io.dat.bits.dataId === ((c.beatsPerLine - 1) * c.dataBits / 128).U)
  }).asUInt

  io.rxRsp.ready := true.B
  io.rxDat.ready := true.B
  when(io.rxRsp.valid) {
    assert(io.rxRsp.bits.txnId < n.U && io.rxRsp.bits.tgtId === p.homeId.U, "Unknown Home response ID")
  }
  when(io.rxDat.valid) {
    assert(io.rxDat.bits.txnId < n.U && io.rxDat.bits.tgtId === p.homeId.U, "Unknown Home data ID")
  }
  when(io.memoryResp.valid)(assert(io.memoryResp.bits.id < n.U, "Unknown backing-memory ID"))
  when(cache.io.response.valid)(assert(cache.io.response.bits.id < n.U, "Unknown Cache operation ID"))

  for (i <- 0 until n) {
    val active       = entries(i).valid
    val r            = entries(i).request
    val set          = r.addr(p.cache.offsetBits + p.cache.setBits - 1, p.cache.offsetBits)
    val lookupResult = cache.io.response.valid && cache.io.response.bits.id === i.U && state(i) === lookupWait
    directory.io.read(i).set := set
    directory.io.read(i).way := Mux(lookupResult, cache.io.response.bits.way, way(i))
    val holders = directory.io.entries(i).sharers
    val owned   = directory.io.entries(i).unique
    val update  = directory.io.update(i)
    update.valid        := false.B
    update.bits.address := directory.io.read(i)
    update.bits.entry   := directory.io.entries(i)
    val mask         = UIntToOH(r.srcId - 1.U, p.agents)
    val exclusive    = r.opcode === Opcode.ReadUnique.U
    val cleanShared  = r.opcode === Opcode.CleanShared.U
    val cleanInvalid = r.opcode === Opcode.CleanInvalid.U
    val makeInvalid  = r.opcode === Opcode.MakeInvalid.U
    val maintenance  = cleanShared || cleanInvalid || makeInvalid
    val writeback    = r.opcode === Opcode.WriteBackFull.U
    val invalidating = victim(i) || cleanInvalid || makeInvalid || exclusive
    val cacheCmd     = cacheArb.io.in(i)
    cacheCmd.valid         := active && (state(i) === lookupReq || state(i) === invalidateReq || state(i) === fillReq || state(
      i
    ) === updateReq || state(i) === restoreReq)
    cacheCmd.bits          := 0.U.asTypeOf(new CacheRequest(p.cache))
    cacheCmd.bits.id       := i.U
    cacheCmd.bits.addr     := Mux(state(i) === invalidateReq || state(i) === restoreReq, workAddress(i), r.addr)
    cacheCmd.bits.way      := way(i)
    cacheCmd.bits.data     := payload(i).asUInt
    cacheCmd.bits.mask     := Fill(p.cache.lineBytes, 1.U(1.W))
    cacheCmd.bits.eligible := Fill(p.cache.ways, 1.U(1.W))
    cacheCmd.bits.metadata := Mux(state(i) === fillReq, 0.U, dirty(i).asUInt)
    cacheCmd.bits.op       := MuxLookup(state(i), CacheOp.Lookup.U)(Seq(
      invalidateReq -> CacheOp.Invalidate.U,
      fillReq       -> CacheOp.Fill.U,
      updateReq     -> CacheOp.Write.U,
      restoreReq    -> CacheOp.Write.U
    ))
    when(cacheCmd.fire) {
      state(i) := MuxLookup(state(i), lookupWait)(Seq(
        invalidateReq -> invalidateWait,
        fillReq       -> fillWait,
        updateReq     -> updateWait,
        restoreReq    -> restoreWait
      ))
    }
    when(cache.io.response.fire && cache.io.response.bits.id === i.U) {
      val result = cache.io.response.bits
      assert(active, "Cache response for a free MSHR")
      when(state(i) === lookupWait) {
        assert(result.available, "Set owner has no replacement candidate")
        way(i)         := result.way
        resident(i)    := result.hit
        victim(i)      := !result.hit || cleanInvalid || makeInvalid
        workAddress(i) := Mux(result.hit, r.addr, result.addr)
        dirty(i)       := result.metadata(0)
        payload(i)     := result.data.asTypeOf(payload(i))
        when(writeback)(state(i) := wbGrant)
          .elsewhen(r.opcode === Opcode.Evict.U) {
            when(result.hit) {
              update.valid              := true.B
              update.bits.entry.sharers := holders & ~mask
              when(owned && (holders & mask).orR) {
                update.valid             := true.B
                update.bits.entry.unique := false.B
              }
            }
            state(i) := complete
          }.elsewhen(maintenance && !result.hit)(state(i) := complete)
          .elsewhen(!result.hit && !result.entryValid)(state(i) := readMemory)
          .otherwise {
            pending(i) := Mux(
              !result.hit || maintenance || exclusive || owned,
              holders,
              0.U
            )
            state(i)   := chooseSnoop
          }
      }.elsewhen(state(i) === invalidateWait) {
        update.valid              := true.B
        update.bits.entry.sharers := 0.U
        update.bits.entry.unique  := false.B
        state(i)                  := Mux(maintenance, complete, readMemory)
      }.elsewhen(state(i) === fillWait) {
        update.valid              := true.B
        update.bits.entry.sharers := 0.U
        update.bits.entry.unique  := false.B
        sent(i)                   := 0.U
        state(i)                  := grantData
      }.elsewhen(state(i) === restoreWait) {
        sent(i)  := 0.U
        state(i) := Mux(maintenance, complete, grantData)
      }.otherwise {
        assert(state(i) === updateWait, "Unexpected Cache response phase")
        sent(i)  := 0.U
        state(i) := Mux(writeback, finished, Mux(maintenance, complete, grantData))
      }
    }

    when(active && state(i) === chooseSnoop) {
      when(pending(i).orR) {
        target(i) := PriorityEncoder(pending(i))
        seen(i)   := 0.U
        state(i)  := sendSnoop
      }.otherwise {
        when(makeInvalid) {
          dirty(i) := false.B
          state(i) := invalidateReq
        }.elsewhen(cleanShared) {
          state(i) := Mux(dirty(i), writeMemory, updateReq)
        }.otherwise {
          state(i) := Mux(victim(i), Mux(dirty(i), writeMemory, invalidateReq), updateReq)
        }
      }
    }
    val snp = snpArb.io.in(i)
    snp.valid                 := active && state(i) === sendSnoop
    snp.bits                  := 0.U.asTypeOf(new DirectedSnoop(c))
    snp.bits.targetNode       := target(i) +& 1.U
    snp.bits.flit.srcId       := p.homeId.U
    snp.bits.flit.txnId       := i.U
    snp.bits.flit.addr        := workAddress(i) >> 3
    snp.bits.flit.opcode      := Mux(
      makeInvalid,
      Opcode.SnpMakeInvalid.U,
      Mux(
        cleanShared,
        Opcode.SnpCleanShared.U,
        Mux(
          victim(i) || cleanInvalid,
          Opcode.SnpCleanInvalid.U,
          Mux(exclusive, Opcode.SnpUnique.U, Opcode.SnpNotSharedDirty.U)
        )
      )
    )
    snp.bits.flit.retToSrc    := Mux(victim(i) || maintenance, 0.U, 1.U)
    snp.bits.flit.doNotGoToSd := 1.U
    when(snp.fire)(state(i)   := waitSnoop)

    when(io.rxRsp.fire && io.rxRsp.bits.txnId === i.U) {
      val rsp = io.rxRsp.bits
      assert(active && rsp.respErr === 0.U, "Unexpected Home response")
      when(state(i) === waitAck) {
        assert(rsp.opcode === Opcode.CompAck.U && rsp.srcId === r.srcId, "Invalid completion acknowledgement")
        state(i) := finished
      }.otherwise {
        assert(
          state(i) === waitSnoop && seen(i) === 0.U && rsp.opcode === Opcode.SnpResp.U && rsp.srcId === target(i) +& 1.U,
          "Invalid snoop response"
        )
        assert(
          !rsp.resp(2) && (rsp.resp(1, 0) === CoherenceState.I.U || (!invalidating && rsp.resp(
            1,
            0
          ) === CoherenceState.SC.U)),
          "Snoop failed to revoke or downgrade permission"
        )
        pending(i)               := pending(i) & ~UIntToOH(target(i), p.agents)
        when(rsp.resp(1, 0) === CoherenceState.I.U) {
          update.valid              := true.B
          update.bits.entry.sharers := holders & ~UIntToOH(target(i), p.agents)
        }
        update.valid             := true.B
        update.bits.entry.unique := false.B
        state(i)                 := chooseSnoop
      }
    }
    when(io.rxDat.fire && io.rxDat.bits.txnId === i.U) {
      val d        = io.rxDat.bits
      assert(active && d.respErr === 0.U && (state(i) === waitSnoop || state(i) === wbData), "Unexpected Home data")
      val b        = if (c.beatsPerLine == 1) 0.U else d.dataId >> log2Ceil(c.dataBits / 128)
      assert(
        VecInit((0 until c.beatsPerLine).map(j => d.dataId === (j * c.dataBits / 128).U)).asUInt.orR,
        "Invalid DataID"
      )
      assert(!(seen(i) & UIntToOH(b, c.beatsPerLine)).orR, "Duplicate data beat")
      when(seen(i).orR)(assert(d.resp === snoopResponse(i), "Inconsistent multi-beat response state"))
      val received = seen(i) | UIntToOH(b, c.beatsPerLine)
      seen(i)          := received
      snoopResponse(i) := d.resp
      payload(i)(b)    := d.data
      when(state(i) === waitSnoop) {
        assert(d.opcode === Opcode.SnpRespData.U && d.srcId === target(i) +& 1.U && d.be.andR, "Invalid snoop data")
        assert(!makeInvalid, "MakeInvalid snoop must not return data")
        assert(
          d.resp(1, 0) === CoherenceState.I.U || (!invalidating && d.resp(1, 0) === CoherenceState.SC.U),
          "Invalid snoop permission"
        )
        dirty(i) := dirty(i) || d.resp(2)
        when(received.andR) {
          pending(i)               := pending(i) & ~UIntToOH(target(i), p.agents)
          when(d.resp(1, 0) === CoherenceState.I.U) {
            update.valid              := true.B
            update.bits.entry.sharers := holders & ~UIntToOH(target(i), p.agents)
          }
          update.valid             := true.B
          update.bits.entry.unique := false.B
          state(i)                 := chooseSnoop
        }
      }.otherwise {
        assert(d.opcode === Opcode.CopyBackWriteData.U && d.srcId === r.srcId, "Invalid CPU writeback")
        when(d.resp === CoherenceState.I.U)(assert(d.be === 0.U, "Cancelled writeback has valid bytes"))
          .otherwise {
            assert(resident(i) && (holders & mask).orR && d.be.andR, "Writeback lacks directory ownership")
          }
        dirty(i) := dirty(i) || d.resp(2)
        when(received.andR) {
          when(d.resp === CoherenceState.I.U)(state(i) := finished)
            .otherwise {
              update.valid              := true.B
              update.bits.entry.sharers := holders & ~mask
              update.bits.entry.unique  := false.B
              state(i)                  := updateReq
            }
        }
      }
    }

    val memory = memArb.io.in(i)
    memory.valid               := active && (state(i) === readMemory || state(i) === writeMemory)
    memory.bits.id             := i.U
    memory.bits.addr           := Mux(state(i) === writeMemory, workAddress(i), r.addr)
    memory.bits.write          := state(i) === writeMemory
    memory.bits.data           := payload(i).asUInt
    memory.bits.mask           := Fill(64, 1.U(1.W))
    when(memory.fire)(state(i) := Mux(state(i) === readMemory, waitRead, waitWrite))
    when(io.memoryResp.fire && io.memoryResp.bits.id === i.U) {
      assert(active && (state(i) === waitRead || state(i) === waitWrite), "Unexpected backing-memory response")
      when(state(i) === waitRead) {
        failed(i)  := io.memoryResp.bits.error
        payload(i) := io.memoryResp.bits.data.asTypeOf(payload(i))
        sent(i)    := 0.U
        state(i)   := Mux(io.memoryResp.bits.error, grantData, fillReq)
      }.otherwise {
        when(io.memoryResp.bits.error) {
          failed(i) := true.B
          state(i)  := restoreReq
        }.otherwise {
          when(cleanShared) {
            dirty(i) := false.B
            state(i) := updateReq
          }.otherwise(state(i) := invalidateReq)
        }
      }
    }
    val rsp = rspArb.io.in(i)
    rsp.valid                  := active && (state(i) === complete || state(i) === wbGrant)
    rsp.bits                   := 0.U.asTypeOf(new ResponseFlit(c))
    rsp.bits.tgtId             := r.srcId
    rsp.bits.srcId             := p.homeId.U
    rsp.bits.txnId             := r.txnId
    rsp.bits.dbid              := i.U
    rsp.bits.respErr           := Mux(failed(i), 3.U, 0.U)
    rsp.bits.opcode            := Mux(state(i) === wbGrant, Opcode.CompDBIDResp.U, Opcode.Comp.U)
    when(rsp.fire) {
      seen(i)  := 0.U
      state(i) := Mux(state(i) === wbGrant, wbData, finished)
    }
    val dat = datArb.io.in(i)
    dat.valid                  := active && state(i) === grantData
    dat.bits                   := 0.U.asTypeOf(new DataFlit(c))
    dat.bits.tgtId             := r.srcId
    dat.bits.srcId             := p.homeId.U
    dat.bits.homeNid           := p.homeId.U
    dat.bits.txnId             := r.txnId
    dat.bits.dbid              := i.U
    dat.bits.opcode            := Opcode.CompData.U
    dat.bits.resp              := Mux(failed(i), CoherenceState.I.U, Mux(exclusive, CoherenceState.UC.U, CoherenceState.SC.U))
    dat.bits.respErr           := Mux(failed(i), 3.U, 0.U)
    dat.bits.dataId            := sent(i) * (c.dataBits / 128).U
    dat.bits.data              := Mux(failed(i), 0.U, payload(i)(if (c.beatsPerLine == 1) 0.U else sent(i)))
    dat.bits.be                := Mux(failed(i), 0.U, Fill(c.bytesPerBeat, 1.U(1.W)))
    when(dat.fire) {
      sent(i) := sent(i) + 1.U
      when(sent(i) === (c.beatsPerLine - 1).U) {
        when(!failed(i)) {
          update.valid              := true.B
          update.bits.entry.sharers := Mux(exclusive, mask, holders | mask)
          update.bits.entry.unique  := exclusive
        }
        state(i) := waitAck
      }
    }
  }
}
