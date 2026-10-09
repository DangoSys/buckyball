package memcore.bus.chi.rnf

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import chisel3.experimental.hierarchy.{instantiable, public}

// Direct-mapped write-back coherent cache agent. The snoop engine is independent
// of the demand miss FSM, so a queued eviction/miss cannot block a Home snoop.
@instantiable
class ChiCache(config: RnfParams, bankIndex: Int) extends Module {
  val p          = config.chi
  val nodeId     = config.nodeId
  val homeId     = config.homeId
  val homeCount  = config.homeCount
  val cacheLines = config.cacheLines / config.banks
  val bankCount  = config.banks
  val txnId      = bankIndex
  require(bankIndex >= 0 && bankIndex < config.banks)
  require(cacheLines >= 2 && isPow2(cacheLines))
  require(nodeId > 0 && nodeId < homeId)
  require(homeId + homeCount <= (1 << p.nodeIdBits))
  require(txnId >= 0 && txnId < 256 && bankCount >= 1 && isPow2(bankCount))
  val mapping    = HomeMapping(homeCount, homeId)

  @public
  val io = IO(new Bundle {
    val access          = Flipped(Decoupled(new CacheAccess(p)))
    val probe           = Option.when(config.probe)(new CacheProbe(p))
    val result          = Decoupled(new CacheResult(config.resultLineBits))
    val chi             = new RequesterPort(p)
    val hits            = Output(UInt(32.W))
    val misses          = Output(UInt(32.W))
    val directory       = Output(Vec(cacheLines, new CacheLineState(p)))
    val dropReservation = Input(Bool())
  })

  val valid           = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val writable        = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val dirty           = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val tags            = Reg(Vec(cacheLines, UInt((p.addressBits - 6).W)))
  val data            = SyncReadMem(cacheLines, UInt(512.W))
  val memoryRead      = WireDefault(false.B)
  val memoryWrite     = WireDefault(false.B)
  val memoryAddress   = WireDefault(0.U(log2Ceil(cacheLines).W))
  val memoryWriteData = WireDefault(0.U(512.W))
  val memoryData      = data.readWrite(memoryAddress, memoryWriteData, memoryRead || memoryWrite, memoryWrite)
  assert(!(memoryRead && memoryWrite), "CHI cache data port conflict")
  val probePending    = RegInit(false.B)
  val probeCommand    = Reg(new CacheProbeRequest(p))
  val probeHit        = Reg(Bool())
  val probeLine       = Reg(UInt(512.W))
  val lookupLine      = Reg(UInt(512.W))
  val lookupFresh     = RegNext(io.access.fire, false.B)
  when(lookupFresh)(lookupLine := memoryData)
  val lookupData  = Mux(lookupFresh, memoryData, lookupLine)
  val fillFailed  = Reg(Bool())
  val fillDoWrite = Reg(Bool())
  def index(addr: UInt): UInt = addr(log2Ceil(cacheLines) + log2Ceil(bankCount) + 5, log2Ceil(bankCount) + 6)
  def tag(addr:   UInt): UInt = addr(p.addressBits - 1, 6)
  val idle :: lookup :: evictReq :: evictWait :: copyback :: getReq :: fill :: fillCommit :: ack :: respond :: Nil =
    Enum(10)
  val state                                                                                                        = RegInit(idle)
  val command                                                                                                      = Reg(new CacheAccess(p))
  val reservation                                                                                                  = RegInit(false.B)
  val reservationAddress                                                                                           = Reg(UInt(p.addressBits.W))
  val reservationWord                                                                                              = Reg(Bool())
  val isLR                                                                                                         = command.atomic === CacheAtomic.LR.U
  val isSC                                                                                                         = command.atomic === CacheAtomic.SC.U
  val modifies                                                                                                     = command.write || (command.atomic >= CacheAtomic.Swap.U && command.atomic <= CacheAtomic.MaxU.U) || isSC
  val reservationMatch                                                                                             = reservation && reservationAddress === command.addr && reservationWord === command.atomicWord
  val victimAddress                                                                                                = Reg(UInt(p.addressBits.W))
  val victimWasDirty                                                                                               = Reg(Bool())
  val copyData                                                                                                     = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val copyResp                                                                                                     = Reg(UInt(3.W))
  val bufferId                                                                                                     = Reg(UInt(p.dbIdBits.W))
  val completionId                                                                                                 = Reg(UInt(p.dbIdBits.W))
  val count                                                                                                        = RegInit(0.U(math.max(1, log2Ceil(p.beatsPerLine)).W))
  val fillData                                                                                                     = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val fillSeen                                                                                                     = RegInit(0.U(p.beatsPerLine.W))
  val fillPermission                                                                                               = Reg(UInt(3.W))
  val fillDbid                                                                                                     = Reg(UInt(p.dbIdBits.W))
  val fillError                                                                                                    = RegInit(false.B)
  val answer                                                                                                       = Reg(new CacheResult(config.resultLineBits))
  val hits                                                                                                         = RegInit(0.U(32.W))
  val misses                                                                                                       = RegInit(0.U(32.W))
  io.hits   := hits
  io.misses := misses
  val ci                                             = index(command.addr)
  val siIdle :: siReadWait :: siRsp :: siData :: Nil = Enum(4)
  val snpState                                       = RegInit(siIdle)
  val snoop                                          = Reg(new SnoopFlit(p))
  val snoopData                                      = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val snoopResult                                    = Reg(UInt(3.W))
  val snoopCount                                     = RegInit(0.U(math.max(1, log2Ceil(p.beatsPerLine)).W))
  // CHI B4.11.1: once fill data starts, a same-line snoop waits for the whole line.
  // Its pending VALID must not block the remaining DAT packets.
  val deferSnoop                                     = (state === fill || state === fillCommit) && (fillSeen.orR || io.chi.rxDat.valid) &&
    tag(io.chi.snp.bits.addr << 3) === tag(command.addr)
  val noSnoop                                        = snpState === siIdle && (!io.chi.snp.valid || deferSnoop)
  io.access.ready := state === idle && noSnoop && !probePending && !io.probe.map(_.req.valid).getOrElse(false.B)

  when(io.access.fire) {
    assert(
      Mux(io.access.bits.atomicWord, io.access.bits.addr(1, 0) === 0.U, io.access.bits.addr(2, 0) === 0.U),
      "Cache client requires naturally aligned accesses"
    )
    assert(io.access.bits.atomic <= CacheAtomic.Fence.U, "Unknown CPU atomic operation")
    when(io.access.bits.atomic =/= CacheAtomic.None.U) {
      assert(!io.access.bits.write && io.access.bits.mask.andR, "Atomic operation requires a full operand")
    }
    command       := io.access.bits
    memoryRead    := true.B
    memoryAddress := index(io.access.bits.addr)
    state         := lookup
  }

  io.probe.foreach { probe =>
    val pi = index(probe.req.bits.addr)
    probe.req.ready := state === idle && snpState === siIdle && !io.chi.snp.valid && !probePending
    val fresh = RegNext(probe.req.fire, false.B)
    when(fresh)(probeLine := memoryData)
    val line = Mux(fresh, memoryData, probeLine)
    probe.complete.valid      := probePending && !probe.cancel && (!probeCommand.write || probe.retire)
    probe.complete.bits.hit   := probeHit
    probe.complete.bits.value := (line >> (probeCommand.addr(5, 3) << 6))(63, 0)
    when(probe.req.fire) {
      probePending  := true.B
      probeCommand  := probe.req.bits
      probeHit      := valid(pi) && tags(pi) === tag(probe.req.bits.addr) && (!probe.req.bits.write || writable(pi))
      memoryRead    := true.B
      memoryAddress := pi
    }
    when(probe.cancel) {
      probePending := false.B
    }.elsewhen(probe.complete.fire) {
      probePending := false.B
      when(probeHit) {
        hits := hits + 1.U
        when(probeCommand.write) {
          val bytes = Wire(Vec(64, UInt(8.W)))
          bytes := line.asTypeOf(bytes)
          for (b <- 0 until 64) {
            when(probeCommand.addr(5, 3) === (b / 8).U && probeCommand.mask(b % 8)) {
              bytes(b) := probeCommand.data((b % 8) * 8 + 7, (b % 8) * 8)
            }
          }
          memoryWrite := true.B
          memoryAddress                   := index(probeCommand.addr)
          memoryWriteData                 := bytes.asUInt
          dirty(index(probeCommand.addr)) := true.B
          reservation                     := false.B
        }
      }
    }
  }

  def oldValue(line: UInt): UInt = {
    val doubleword = (line >> (command.addr(5, 3) << 6))(63, 0)
    val word       = Mux(command.addr(2), doubleword(63, 32), doubleword(31, 0))
    Mux(command.atomicWord, Cat(Fill(32, word(31)), word), doubleword)
  }

  def merge(line: UInt): UInt = {
    val old     = (line >> (command.addr(5, 3) << 6))(63, 0)
    val oldWord = Mux(command.addr(2), old(63, 32), old(31, 0))
    def operation(lhs: UInt, rhs: UInt): UInt = MuxLookup(command.atomic, rhs)(Seq(
      CacheAtomic.Add.U  -> (lhs + rhs),
      CacheAtomic.Xor.U  -> (lhs ^ rhs),
      CacheAtomic.And.U  -> (lhs & rhs),
      CacheAtomic.Or.U   -> (lhs | rhs),
      CacheAtomic.Min.U  -> Mux(lhs.asSInt < rhs.asSInt, lhs, rhs),
      CacheAtomic.Max.U  -> Mux(lhs.asSInt > rhs.asSInt, lhs, rhs),
      CacheAtomic.MinU.U -> Mux(lhs < rhs, lhs, rhs),
      CacheAtomic.MaxU.U -> Mux(lhs > rhs, lhs, rhs)
    ))
    val wordValue = operation(oldWord, command.data(31, 0))
    val value = Mux(
      command.atomicWord,
      Mux(command.addr(2), Cat(wordValue, old(31, 0)), Cat(old(63, 32), wordValue)),
      operation(old, command.data)
    )
    val bytes = Wire(Vec(64, UInt(8.W)))
    bytes := line.asTypeOf(bytes)
    for (b <- 0 until 64) {
      val selected = !command.atomicWord || command.addr(2) === (b % 8 / 4).U
      when(command.addr(5, 3) === (b / 8).U && command.mask(b % 8) && selected) {
        bytes(b) := value((b % 8) * 8 + 7, (b % 8) * 8)
      }
    }
    bytes.asUInt
  }

  when(state === lookup && snpState === siIdle) {
    val hit = valid(ci) && tags(ci) === tag(command.addr)
    when(command.atomic === CacheAtomic.Fence.U || (isSC && !reservationMatch)) {
      answer.data                        := Mux(isSC, 1.U, 0.U)
      answer.error                       := false.B
      if (config.lineResult) answer.line := 0.U
      when(isSC)(reservation             := false.B)
      state                              := respond
    }.elsewhen(hit && (!modifies || writable(ci))) {
      hits                               := hits + 1.U
      answer.data                        := Mux(isSC, 0.U, oldValue(lookupData))
      answer.error                       := false.B
      if (config.lineResult) answer.line := lookupData
      when(modifies) {
        memoryWrite := true.B; memoryAddress := ci; memoryWriteData := merge(lookupData); dirty(ci) := true.B;
        reservation := false.B
      }
      when(isLR) { reservation := true.B; reservationAddress := command.addr; reservationWord := command.atomicWord }
      state                              := respond
    }.otherwise {
      misses            := misses + 1.U
      when(valid(ci) && !hit) {
        victimAddress  := tags(ci) << 6
        victimWasDirty := dirty(ci)
        copyData       := lookupData.asTypeOf(copyData)
        state          := evictReq
      }.otherwise(state := getReq)
    }
  }

  io.chi.req.valid           := state === evictReq || state === getReq
  io.chi.req.bits            := 0.U.asTypeOf(new RequestFlit(p))
  io.chi.req.bits.srcId      := nodeId.U
  io.chi.req.bits.txnId      := txnId.U
  io.chi.req.bits.tgtId      := Mux(state === evictReq, mapping.node(victimAddress), mapping.node(command.addr))
  io.chi.req.bits.addr       := Mux(state === evictReq, victimAddress, (command.addr >> 6) << 6)
  io.chi.req.bits.size       := 6.U
  io.chi.req.bits.opcode     := Mux(
    state === evictReq,
    Mux(victimWasDirty, Opcode.WriteBackFull.U, Opcode.Evict.U),
    Mux(modifies, Opcode.ReadUnique.U, Opcode.ReadNotSharedDirty.U)
  )
  io.chi.req.bits.snpAttr    := 1.U
  io.chi.req.bits.memAttr    := "b1100".U
  io.chi.req.bits.allowRetry := 1.U
  io.chi.req.bits.expCompAck := (state === getReq).asUInt
  when(io.chi.req.fire) {
    state     := Mux(state === evictReq, evictWait, fill)
    fillSeen  := 0.U
    fillError := false.B
  }

  io.chi.rxRsp.ready := state === evictWait && noSnoop
  when(io.chi.rxRsp.fire) {
    val r            = io.chi.rxRsp.bits
    assert(
      r.tgtId === nodeId.U && r.srcId === mapping.node(victimAddress) && r.txnId === txnId.U && r.respErr === 0.U,
      "Unexpected eviction completion"
    )
    val vi           = index(victimAddress)
    val stillPresent = valid(vi) && tags(vi) === tag(victimAddress)
    when(victimWasDirty) {
      assert(r.opcode === Opcode.CompDBIDResp.U, "WriteBackFull requires CompDBIDResp")
      copyResp := Mux(
        !stillPresent,
        CoherenceState.I.U,
        Mux(
          dirty(vi),
          (CoherenceState.UC | CoherenceState.PassDirty).U,
          Mux(writable(vi), CoherenceState.UC.U, CoherenceState.SC.U)
        )
      )
      bufferId := r.dbid
      count    := 0.U
      state    := copyback
    }.otherwise {
      assert(r.opcode === Opcode.Comp.U, "Evict requires Comp")
      state := getReq
    }
    valid(vi) := false.B
    writable(vi)                                                     := false.B
    dirty(vi)                                                        := false.B
    when(tag(reservationAddress) === tag(victimAddress))(reservation := false.B)
  }

  io.chi.rxDat.ready := state === fill && noSnoop
  when(io.chi.rxDat.fire) {
    val d        = io.chi.rxDat.bits
    assert(
      d.opcode === Opcode.CompData.U && d.srcId === mapping.node(command.addr) &&
        d.tgtId === nodeId.U && d.txnId === txnId.U && d.homeNid === mapping.node(command.addr),
      "Unexpected cache fill"
    )
    val b        = if (p.beatsPerLine == 1) 0.U(0.W) else d.dataId >> log2Ceil(p.dataBits / 128)
    assert(
      VecInit((0 until p.beatsPerLine).map(i => d.dataId === (i * p.dataBits / 128).U)).asUInt.orR,
      "Invalid fill DataID"
    )
    assert(!(fillSeen & UIntToOH(b, p.beatsPerLine)).orR, "Duplicate fill DataID")
    val received = fillSeen | UIntToOH(b, p.beatsPerLine)
    val nextData = WireInit(fillData)
    nextData(b)    := d.data
    assert(d.dbid < (BigInt(1) << p.dbIdBits).U, "Fill DBID exceeds completion ID width")
    when(fillSeen.orR) {
      assert(d.dbid === fillDbid && d.resp === fillPermission, "Inconsistent multi-beat fill metadata")
    }
    fillPermission := d.resp
    fillDbid       := d.dbid
    fillData       := nextData
    fillSeen       := received
    fillError      := fillError || d.respErr =/= 0.U
    completionId   := d.dbid(p.dbIdBits - 1, 0)
    when(received.andR) {
      val failed = fillError || d.respErr =/= 0.U
      fillFailed                         := failed
      fillDoWrite                        := modifies && (!isSC || reservationMatch)
      answer.data                        := Mux(failed, 0.U, Mux(isSC, !reservationMatch, oldValue(nextData.asUInt)))
      if (config.lineResult) answer.line := nextData.asUInt
      when(modifies)(reservation         := false.B)
      answer.error                       := failed
      state                              := fillCommit
    }
  }
  when(state === fillCommit && noSnoop) {
    when(!fillFailed) {
      assert(
        fillPermission === Mux(modifies, CoherenceState.UC.U, CoherenceState.SC.U),
        "Home granted unexpected cache permission"
      )
      valid(ci) := true.B
      val doWrite = fillDoWrite
      writable(ci)    := modifies
      dirty(ci)       := doWrite
      tags(ci)        := tag(command.addr)
      memoryWrite     := true.B
      memoryAddress   := ci
      memoryWriteData := Mux(doWrite, merge(fillData.asUInt), fillData.asUInt)
      when(isLR) { reservation := true.B; reservationAddress := command.addr; reservationWord := command.atomicWord }
    }
    state := ack
  }

  io.chi.snp.ready := snpState === siIdle && !deferSnoop && state =/= lookup && !probePending
  when(io.chi.snp.fire) {
    val s           = io.chi.snp.bits
    val address     = s.addr << 3
    val i           = index(address)
    val hit         = valid(i) && tags(i) === tag(address)
    val discard     = s.opcode === Opcode.SnpMakeInvalid.U
    val cleanShared = s.opcode === Opcode.SnpCleanShared.U
    val invalidate  = s.opcode === Opcode.SnpUnique.U || s.opcode === Opcode.SnpCleanInvalid.U || discard
    // Other harts' read-only traffic must not indefinitely defeat constrained LR/SC.
    when(reservation && tag(reservationAddress) === tag(address) && invalidate)(reservation := false.B)
    assert(
      s.srcId === mapping.node(address) && s.pas === 0.U && s.fwdNid === 0.U && s.fwdTxnId === 0.U &&
        (invalidate || cleanShared || s.opcode === Opcode.SnpNotSharedDirty.U),
      "Unsupported cache snoop"
    )
    when(cleanShared || s.opcode === Opcode.SnpCleanInvalid.U || discard) {
      assert(!s.retToSrc.asBool, "Cache maintenance snoop requires RetToSrc zero")
    }
    snoop                                                                                   := s
    snoopCount                                                                              := 0.U
    val finalState = Mux(!hit || invalidate, CoherenceState.I.U, CoherenceState.SC.U)
    snoopResult := finalState | Mux(hit && dirty(i) && !discard, CoherenceState.PassDirty.U, 0.U)
    val readsData = hit && !discard && (dirty(i) || s.retToSrc.asBool)
    snpState := Mux(readsData, siReadWait, siRsp)
    when(readsData) { memoryRead := true.B; memoryAddress := i }
    when(hit) {
      writable(i)               := false.B
      dirty(i)                  := false.B
      when(invalidate)(valid(i) := false.B)
    }
  }

  when(snpState === siReadWait) {
    snoopData := memoryData.asTypeOf(snoopData)
    snpState  := siData
  }

  io.chi.txRsp.valid       := snpState === siRsp || state === ack
  io.chi.txRsp.bits        := 0.U.asTypeOf(new ResponseFlit(p))
  io.chi.txRsp.bits.srcId  := nodeId.U
  io.chi.txRsp.bits.tgtId  := Mux(snpState === siRsp, snoop.srcId, mapping.node(command.addr))
  io.chi.txRsp.bits.txnId  := Mux(snpState === siRsp, snoop.txnId, completionId)
  io.chi.txRsp.bits.opcode := Mux(snpState === siRsp, Opcode.SnpResp.U, Opcode.CompAck.U)
  io.chi.txRsp.bits.resp   := Mux(snpState === siRsp, snoopResult, 0.U)
  when(io.chi.txRsp.fire) {
    when(snpState === siRsp)(snpState := siIdle).otherwise(state := respond)
  }

  io.chi.txDat.valid       := snpState === siData || state === copyback
  io.chi.txDat.bits        := 0.U.asTypeOf(new DataFlit(p))
  io.chi.txDat.bits.srcId  := nodeId.U
  io.chi.txDat.bits.tgtId  := Mux(snpState === siData, snoop.srcId, mapping.node(victimAddress))
  io.chi.txDat.bits.txnId  := Mux(snpState === siData, snoop.txnId, bufferId)
  io.chi.txDat.bits.opcode := Mux(snpState === siData, Opcode.SnpRespData.U, Opcode.CopyBackWriteData.U)
  io.chi.txDat.bits.resp   := Mux(snpState === siData, snoopResult, copyResp)
  val outputBeat = Mux(snpState === siData, snoopCount, count)
  val b          = if (p.beatsPerLine == 1) 0.U(0.W) else outputBeat
  io.chi.txDat.bits.dataId             := outputBeat * (p.dataBits / 128).U
  io.chi.txDat.bits.data               := Mux(snpState === siData, snoopData(b), Mux(copyResp === CoherenceState.I.U, 0.U, copyData(b)))
  io.chi.txDat.bits.be                 := Mux(
    snpState === siData || copyResp =/= CoherenceState.I.U,
    Fill(p.bytesPerBeat, 1.U(1.W)),
    0.U
  )
  when(io.chi.txDat.fire) {
    when(snpState === siData) {
      snoopCount                                           := snoopCount + 1.U
      when(snoopCount === (p.beatsPerLine - 1).U)(snpState := siIdle)
    }.otherwise {
      count                                        := count + 1.U
      when(count === (p.beatsPerLine - 1).U)(state := getReq)
    }
  }
  io.result.valid                      := state === respond
  io.result.bits                       := answer
  when(io.result.fire)(state           := idle)
  when(io.dropReservation)(reservation := false.B)
  for (i <- 0 until cacheLines) {
    io.directory(i).valid    := valid(i)
    io.directory(i).writable := writable(i)
    io.directory(i).line     := tags(i)
    assert(!dirty(i) || (valid(i) && writable(i)), "Dirty cache line without Unique ownership")
    assert(!writable(i) || valid(i), "Invalid cache line has write permission")
  }
}
