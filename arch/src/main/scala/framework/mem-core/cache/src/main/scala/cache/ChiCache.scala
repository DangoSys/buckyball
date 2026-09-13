package memcore.memory.cache

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import memcore.memory.coherence._

// Direct-mapped write-back coherent cache agent. The snoop engine is independent
// of the demand miss FSM, so a queued eviction/miss cannot block a Home snoop.
class ChiCache(
  p:          ChiParams,
  nodeId:     Int,
  cacheLines: Int = 4,
  homeId:     Int = 64,
  homeCount:  Int = 1,
  txnId:      Int = 0,
  bankCount:  Int = 1)
    extends Module {
  require(cacheLines >= 2 && isPow2(cacheLines))
  require(nodeId > 0 && nodeId < homeId)
  require(homeId + homeCount <= (1 << p.nodeIdBits))
  require(txnId >= 0 && txnId < 256 && bankCount >= 1 && isPow2(bankCount))
  val mapping = HomeMapping(homeCount, homeId)

  val io = IO(new Bundle {
    val access          = Flipped(Decoupled(new CacheAccess(p)))
    val result          = Decoupled(new CacheResult)
    val chi             = new ChiRequesterPort(p)
    val hits            = Output(UInt(32.W))
    val misses          = Output(UInt(32.W))
    val directory       = Output(Vec(cacheLines, new CacheLineState(p)))
    val dropReservation = Input(Bool())
  })

  val valid    = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val writable = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val dirty    = RegInit(VecInit(Seq.fill(cacheLines)(false.B)))
  val tags     = Reg(Vec(cacheLines, UInt((p.addressBits - 6).W)))
  val data     = Mem(cacheLines, UInt(512.W))
  def index(addr: UInt): UInt = addr(log2Ceil(cacheLines) + log2Ceil(bankCount) + 5, log2Ceil(bankCount) + 6)
  def tag(addr:   UInt): UInt = addr(p.addressBits - 1, 6)
  val idle :: lookup :: evictReq :: evictWait :: copyback :: getReq :: fill :: ack :: respond :: Nil = Enum(9)
  val state                                                                                          = RegInit(idle)
  val command                                                                                        = Reg(new CacheAccess(p))
  val reservation                                                                                    = RegInit(false.B)
  val reservationAddress                                                                             = Reg(UInt(p.addressBits.W))
  val isLR                                                                                           = command.atomic === CacheAtomic.LR.U
  val isSC                                                                                           = command.atomic === CacheAtomic.SC.U
  val modifies                                                                                       = command.write || (command.atomic >= CacheAtomic.Swap.U && command.atomic <= CacheAtomic.MaxU.U) || isSC
  val reservationMatch                                                                               = reservation && reservationAddress === command.addr
  val victimAddress                                                                                  = Reg(UInt(p.addressBits.W))
  val victimWasDirty                                                                                 = Reg(Bool())
  val copyData                                                                                       = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val copyResp                                                                                       = Reg(UInt(3.W))
  val bufferId                                                                                       = Reg(UInt(8.W))
  val completionId                                                                                   = Reg(UInt(8.W))
  val count                                                                                          = RegInit(0.U(math.max(1, log2Ceil(p.beatsPerLine)).W))
  val fillData                                                                                       = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val fillSeen                                                                                       = RegInit(0.U(p.beatsPerLine.W))
  val fillError                                                                                      = RegInit(false.B)
  val answer                                                                                         = Reg(new CacheResult)
  val hits                                                                                           = RegInit(0.U(32.W))
  val misses                                                                                         = RegInit(0.U(32.W))
  io.hits   := hits
  io.misses := misses
  val ci                               = index(command.addr)
  val siIdle :: siRsp :: siData :: Nil = Enum(3)
  val snpState                         = RegInit(siIdle)
  val snoop                            = Reg(new ChiSnp(p))
  val snoopData                        = Reg(Vec(p.beatsPerLine, UInt(p.dataBits.W)))
  val snoopResult                      = Reg(UInt(3.W))
  val snoopCount                       = RegInit(0.U(math.max(1, log2Ceil(p.beatsPerLine)).W))
  val noSnoop                          = snpState === siIdle && !io.chi.snp.valid
  io.access.ready := state === idle && noSnoop
  when(io.access.fire) {
    assert(io.access.bits.addr(2, 0) === 0.U, "Cache client requires aligned 64-bit accesses")
    assert(io.access.bits.atomic <= CacheAtomic.Fence.U, "Unknown CPU atomic operation")
    when(io.access.bits.atomic =/= 0.U) {
      assert(!io.access.bits.write && io.access.bits.mask.andR, "Atomic operation requires a full 64-bit operand")
    }
    command := io.access.bits
    state   := lookup
  }

  def merge(line: UInt): UInt = {
    val old     = (line >> (command.addr(5, 3) << 6))(63, 0)
    val operand = command.data
    val value   = MuxLookup(command.atomic, operand)(Seq(
      CacheAtomic.Add.U  -> (old + operand),
      CacheAtomic.Xor.U  -> (old ^ operand),
      CacheAtomic.And.U  -> (old & operand),
      CacheAtomic.Or.U   -> (old | operand),
      CacheAtomic.Min.U  -> Mux(old.asSInt < operand.asSInt, old, operand),
      CacheAtomic.Max.U  -> Mux(old.asSInt > operand.asSInt, old, operand),
      CacheAtomic.MinU.U -> Mux(old < operand, old, operand),
      CacheAtomic.MaxU.U -> Mux(old > operand, old, operand)
    ))
    val bytes   = Wire(Vec(64, UInt(8.W)))
    bytes := line.asTypeOf(bytes)
    for (b <- 0 until 64) {
      when(command.addr(5, 3) === (b / 8).U && command.mask(b % 8)) {
        bytes(b) := value((b % 8) * 8 + 7, (b % 8) * 8)
      }
    }
    bytes.asUInt
  }

  when(state === lookup && noSnoop) {
    val hit = valid(ci) && tags(ci) === tag(command.addr)
    when(command.atomic === CacheAtomic.Fence.U || (isSC && !reservationMatch)) {
      answer.data            := Mux(isSC, 1.U, 0.U)
      answer.error           := false.B
      when(isSC)(reservation := false.B)
      state                  := respond
    }.elsewhen(hit && (!modifies || writable(ci))) {
      hits         := hits + 1.U
      answer.data  := Mux(isSC, 0.U, (data(ci) >> (command.addr(5, 3) << 6))(63, 0))
      answer.error := false.B
      when(modifies) {
        data(ci) := merge(data(ci)); dirty(ci) := true.B; reservation := false.B
      }
      when(isLR) { reservation := true.B; reservationAddress := command.addr }
      state        := respond
    }.otherwise {
      misses            := misses + 1.U
      when(valid(ci) && !hit) {
        victimAddress  := tags(ci) << 6
        victimWasDirty := dirty(ci)
        state          := evictReq
      }.otherwise(state := getReq)
    }
  }

  io.chi.req.valid           := state === evictReq || state === getReq
  io.chi.req.bits            := 0.U.asTypeOf(new ChiReq(p))
  io.chi.req.bits.srcId      := nodeId.U
  io.chi.req.bits.txnId      := txnId.U
  io.chi.req.bits.tgtId      := Mux(state === evictReq, mapping.node(victimAddress), mapping.node(command.addr))
  io.chi.req.bits.addr       := Mux(state === evictReq, victimAddress, (command.addr >> 6) << 6)
  io.chi.req.bits.size       := 6.U
  io.chi.req.bits.opcode     := Mux(
    state === evictReq,
    Mux(victimWasDirty, ChiOpcode.WriteBackFull.U, ChiOpcode.Evict.U),
    Mux(modifies, ChiOpcode.ReadUnique.U, ChiOpcode.ReadNotSharedDirty.U)
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
      assert(r.opcode === ChiOpcode.CompDBIDResp.U, "WriteBackFull requires CompDBIDResp")
      copyData := data(vi).asTypeOf(copyData)
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
      assert(r.opcode === ChiOpcode.Comp.U, "Evict requires Comp")
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
      d.opcode === ChiOpcode.CompData.U && d.srcId === mapping.node(command.addr) &&
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
    nextData(b)  := d.data
    fillData     := nextData
    fillSeen     := received
    fillError    := fillError || d.respErr =/= 0.U
    completionId := d.dbid
    when(received.andR) {
      val failed = fillError || d.respErr =/= 0.U
      when(!failed) {
        assert(
          d.resp === Mux(modifies, CoherenceState.UC.U, CoherenceState.SC.U),
          "Home granted unexpected cache permission"
        )
        valid(ci) := true.B
        val doWrite = modifies && (!isSC || reservationMatch)
        writable(ci) := modifies
        dirty(ci)    := doWrite
        tags(ci)     := tag(command.addr)
        data(ci)     := Mux(doWrite, merge(nextData.asUInt), nextData.asUInt)
        when(isLR) { reservation := true.B; reservationAddress := command.addr }
      }
      answer.data := Mux(failed, 0.U, Mux(isSC, !reservationMatch, (nextData.asUInt >> (command.addr(5, 3) << 6))(63, 0)))
      when(modifies)(reservation := false.B)
      answer.error               := failed
      state                      := ack
    }
  }

  io.chi.snp.ready := snpState === siIdle
  when(io.chi.snp.fire) {
    val s          = io.chi.snp.bits
    val address    = s.addr << 3
    val i          = index(address)
    val hit        = valid(i) && tags(i) === tag(address)
    val invalidate = s.opcode === ChiOpcode.SnpUnique.U || s.opcode === ChiOpcode.SnpCleanInvalid.U
    // Other harts' read-only traffic must not indefinitely defeat constrained LR/SC.
    when(reservation && tag(reservationAddress) === tag(address) && invalidate)(reservation := false.B)
    assert(
      s.srcId === mapping.node(address) && (invalidate || s.opcode === ChiOpcode.SnpNotSharedDirty.U),
      "Unsupported cache snoop"
    )
    snoop                                                                                   := s
    snoopData                                                                               := data(i).asTypeOf(snoopData)
    snoopCount                                                                              := 0.U
    val finalState = Mux(!hit || invalidate, CoherenceState.I.U, CoherenceState.SC.U)
    snoopResult := finalState | Mux(hit && dirty(i), CoherenceState.PassDirty.U, 0.U)
    snpState    := Mux(hit && (dirty(i) || s.retToSrc.asBool), siData, siRsp)
    when(hit) {
      writable(i)               := false.B
      dirty(i)                  := false.B
      when(invalidate)(valid(i) := false.B)
    }
  }

  io.chi.txRsp.valid       := snpState === siRsp || state === ack
  io.chi.txRsp.bits        := 0.U.asTypeOf(new ChiRsp(p))
  io.chi.txRsp.bits.srcId  := nodeId.U
  io.chi.txRsp.bits.tgtId  := Mux(snpState === siRsp, snoop.srcId, mapping.node(command.addr))
  io.chi.txRsp.bits.txnId  := Mux(snpState === siRsp, snoop.txnId, completionId)
  io.chi.txRsp.bits.opcode := Mux(snpState === siRsp, ChiOpcode.SnpResp.U, ChiOpcode.CompAck.U)
  io.chi.txRsp.bits.resp   := Mux(snpState === siRsp, snoopResult, 0.U)
  when(io.chi.txRsp.fire) {
    when(snpState === siRsp)(snpState := siIdle).otherwise(state := respond)
  }

  io.chi.txDat.valid       := snpState === siData || state === copyback
  io.chi.txDat.bits        := 0.U.asTypeOf(new ChiDat(p))
  io.chi.txDat.bits.srcId  := nodeId.U
  io.chi.txDat.bits.tgtId  := Mux(snpState === siData, snoop.srcId, mapping.node(victimAddress))
  io.chi.txDat.bits.txnId  := Mux(snpState === siData, snoop.txnId, bufferId)
  io.chi.txDat.bits.opcode := Mux(snpState === siData, ChiOpcode.SnpRespData.U, ChiOpcode.CopyBackWrData.U)
  io.chi.txDat.bits.resp   := Mux(snpState === siData, snoopResult, copyResp)
  val outputBeat = Mux(snpState === siData, snoopCount, count)
  val b          = if (p.beatsPerLine == 1) 0.U(0.W) else outputBeat
  io.chi.txDat.bits.dataId             := outputBeat * (p.dataBits / 128).U
  io.chi.txDat.bits.data               := Mux(snpState === siData, snoopData(b), copyData(b))
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
