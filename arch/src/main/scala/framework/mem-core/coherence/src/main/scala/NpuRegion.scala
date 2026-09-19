package memcore.memory.coherence

import chisel3._
import chisel3.util._
import memcore.bus.chi._

class RegionRequest(p: Params) extends Bundle {
  val addr  = UInt(p.addressBits.W)
  val lines = UInt(32.W)
  val write = Bool()
}

class RegionEntry(p: Params) extends Bundle {
  val valid     = Bool()
  val published = Bool()
  val base      = UInt(p.addressBits.W)
  val end       = UInt((p.addressBits + 1).W)
  val write     = Bool()
}

// Coarse-grain ownership is an admission protocol above CHI, not a larger CHI
// cache line. Reserve before sweeping CPU caches; publish only after all CMOs
// complete. Distinct NPU regions can be active concurrently.
class RegionDirectory(p: Params, clients: Int, memoryLines: Int) extends Module {
  require(clients >= 1)

  val io = IO(new Bundle {
    val claim   = Vec(clients, Flipped(Decoupled(new RegionRequest(p))))
    val publish = Input(Vec(clients, Bool()))
    val release = Input(Vec(clients, Bool()))
    val entries = Output(Vec(clients, new RegionEntry(p)))
  })

  val entries = RegInit(VecInit(Seq.fill(clients)(0.U.asTypeOf(new RegionEntry(p)))))
  io.entries := entries
  val arb = Module(new RRArbiter(new RegionRequest(p), clients))
  arb.io.out.ready := true.B
  for (i <- 0 until clients) {
    val r        = io.claim(i).bits
    val end      = r.addr +& (r.lines << 6)
    val conflict = entries.map(e => e.valid && r.addr < e.end && end > e.base).reduce(_ || _)
    arb.io.in(i).valid := io.claim(i).valid && !entries(i).valid && !conflict
    arb.io.in(i).bits  := r
    io.claim(i).ready  := arb.io.in(i).ready && !entries(i).valid && !conflict
    when(io.claim(i).valid) {
      assert(!entries(i).valid, "NPU attempted a second live region claim")
      assert(r.lines =/= 0.U && r.addr(5, 0) === 0.U && end <= (BigInt(memoryLines) * 64).U, "Invalid NPU region bounds")
    }
    when(arb.io.in(i).fire) {
      entries(i).valid     := true.B
      entries(i).published := false.B
      entries(i).base      := r.addr
      entries(i).end       := end
      entries(i).write     := r.write
    }
    when(io.publish(i)) {
      assert(entries(i).valid && !entries(i).published, "Invalid region publish")
      entries(i).published := true.B
    }
    when(io.release(i)) {
      assert(entries(i).valid, "Invalid region release")
      entries(i).valid     := false.B
      entries(i).published := false.B
    }
  }
  for {
    a    <- 0 until clients
    b    <- a + 1 until clients
  } {
    when(entries(a).valid && entries(b).valid) {
      assert(entries(a).end <= entries(b).base || entries(b).end <= entries(a).base, "Overlapping NPU reservations")
    }
  }
}

object NpuOperation {
  val AcquireRead  = 0
  val AcquireWrite = 1
  val Read         = 2
  val Write        = 3
  val Release      = 4
}

class NpuCommand(p: Params) extends Bundle {
  val op    = UInt(3.W)
  val addr  = UInt(p.addressBits.W)
  val lines = UInt(32.W)
}

class NpuBeat(p: Params) extends Bundle {
  val data = UInt(p.dataBits.W)
  val mask = UInt(p.bytesPerBeat.W)
  val last = Bool()
}

// Region ownership persists across commands until Release. The ring holds complete
// lines, so independently completing CHI transactions never reorder the Ball stream.
class NpuRegionAgent(
  p:         Params,
  nodeId:    Int,
  homeId:    Int = 64,
  homeCount: Int = 1,
  slots:     Int = 4)
    extends Module {
  require(homeCount >= 1 && isPow2(homeCount))
  require(homeId >= 0 && homeId + homeCount <= (1 << p.nodeIdBits))
  require(nodeId >= 1 && nodeId < homeId)
  require(slots >= 1 && slots <= 256)
  private val slotBits = math.max(1, log2Ceil(slots))
  private val beatBits = math.max(1, log2Ceil(p.beatsPerLine))

  val io = IO(new Bundle {
    val command     = Flipped(Decoupled(new NpuCommand(p)))
    val write       = Flipped(Decoupled(new NpuBeat(p)))
    val read        = Decoupled(new NpuBeat(p))
    val done        = Decoupled(Bool()) // true means an error occurred
    val claim       = Decoupled(new RegionRequest(p))
    val publish     = Output(Bool())
    val release     = Output(Bool())
    val chi         = new RequesterPort(p)
    val outstanding = Output(UInt(log2Ceil(slots + 1).W))
  })

  val Seq(idle, claiming, executing, respond) = Enum(4)
  val state                                   = RegInit(idle)
  val command                                 = Reg(new NpuCommand(p))
  val leased                                  = RegInit(false.B)
  val leaseWrite                              = Reg(Bool())
  val leaseBase                               = Reg(UInt(p.addressBits.W))
  val leaseEnd                                = Reg(UInt((p.addressBits + 1).W))
  val error                                   = RegInit(false.B)
  val acquire                                 = command.op === NpuOperation.AcquireRead.U || command.op === NpuOperation.AcquireWrite.U
  val reading                                 = command.op === NpuOperation.Read.U
  val writing                                 = command.op === NpuOperation.Write.U

  val Seq(free, filling, requestReady, waiting, sending, waitComp, complete) = Enum(7)
  val slotState                                                              = RegInit(VecInit(Seq.fill(slots)(free)))
  val addresses                                                              = Reg(Vec(slots, UInt(p.addressBits.W)))
  val homes                                                                  = Reg(Vec(slots, UInt(p.nodeIdBits.W)))
  val dbids                                                                  = Reg(Vec(slots, UInt(p.dbIdBits.W)))
  val payload                                                                = Reg(Vec(slots, Vec(p.beatsPerLine, UInt(p.dataBits.W))))
  val masks                                                                  = Reg(Vec(slots, Vec(p.beatsPerLine, UInt(p.bytesPerBeat.W))))
  val seen                                                                   = RegInit(VecInit(Seq.fill(slots)(0.U(p.beatsPerLine.W))))
  val sendBeat                                                               = RegInit(VecInit(Seq.fill(slots)(0.U(beatBits.W))))
  val inFlight                                                               = RegInit(VecInit(Seq.fill(slots)(false.B)))
  io.outstanding := PopCount(inFlight)
  val allocate       = RegInit(0.U(slotBits.W))
  val retire         = RegInit(0.U(slotBits.W))
  val allocatedLines = RegInit(0.U(32.W))
  val retiredLines   = RegInit(0.U(32.W))
  val nextAddress    = Reg(UInt(p.addressBits.W))
  val collectBeat    = RegInit(0.U(beatBits.W))
  val readBeat       = RegInit(0.U(beatBits.W))
  def nextSlot(index:  UInt): UInt = Mux(index === (slots - 1).U, 0.U, index + 1.U)
  def homeFor(addr:    UInt): UInt = homeId.U + ((addr >> 6) & (homeCount - 1).U)
  def slotFor(tag:     UInt): UInt = if (slots == 1) 0.U(0.W) else tag(slotBits - 1, 0)
  def beatIndex(index: UInt): UInt = if (p.beatsPerLine == 1) 0.U(0.W) else index

  io.publish       := false.B
  io.release       := false.B
  io.command.ready := state === idle
  when(io.command.fire) {
    val c        = io.command.bits
    val claim    = c.op === NpuOperation.AcquireRead.U || c.op === NpuOperation.AcquireWrite.U
    val transfer = c.op === NpuOperation.Read.U || c.op === NpuOperation.Write.U
    assert(claim || transfer || c.op === NpuOperation.Release.U, "Unknown NPU operation")
    assert(slotState.map(_ === free).reduce(_ && _) && !inFlight.asUInt.orR, "NPU command crossed a live transaction")
    command        := c
    allocate       := 0.U
    retire         := 0.U
    allocatedLines := 0.U
    retiredLines   := 0.U
    nextAddress    := c.addr
    collectBeat    := 0.U
    readBeat       := 0.U
    error          := false.B
    when(claim) {
      assert(!leased, "NPU already owns a region")
      state := claiming
    }.elsewhen(transfer) {
      assert(
        leased && c.lines =/= 0.U && c.addr(5, 0) === 0.U && c.addr >= leaseBase &&
          (c.addr +& (c.lines << 6)) <= leaseEnd,
        "NPU transfer outside owned region"
      )
      when(c.op === NpuOperation.Write.U)(assert(leaseWrite, "Write through read-only NPU lease"))
      state := executing
    }.otherwise {
      assert(leased, "NPU released an unowned region")
      io.release := true.B
      leased     := false.B
      state      := respond
    }
  }

  io.claim.valid      := state === claiming
  io.claim.bits.addr  := command.addr
  io.claim.bits.lines := command.lines
  io.claim.bits.write := command.op === NpuOperation.AcquireWrite.U
  when(io.claim.fire) {
    leaseBase  := command.addr
    leaseEnd   := command.addr +& (command.lines << 6)
    leaseWrite := command.op === NpuOperation.AcquireWrite.U
    state      := executing
  }

  // Non-write commands allocate one line per cycle. Writes assemble consecutive
  // stream beats locally while already assembled lines make progress in CHI.
  val canAllocate = state === executing && allocatedLines < command.lines && slotState(allocate) === free
  when(canAllocate && !writing) {
    addresses(allocate) := nextAddress
    homes(allocate)     := homeFor(nextAddress)
    seen(allocate)      := 0.U
    slotState(allocate) := requestReady
    nextAddress         := nextAddress + 64.U
    allocate            := nextSlot(allocate)
    allocatedLines      := allocatedLines + 1.U
  }
  io.write.ready := state === executing && writing && allocatedLines < command.lines &&
    (slotState(allocate) === free || slotState(allocate) === filling)
  when(io.write.fire) {
    val lastBeat = collectBeat === (p.beatsPerLine - 1).U
    assert(
      io.write.bits.last === (allocatedLines === command.lines - 1.U && lastBeat),
      "NPU stream LAST does not match command length"
    )
    when(collectBeat === 0.U) {
      assert(slotState(allocate) === free, "NPU write allocation overwrote a live line")
      addresses(allocate) := nextAddress
      homes(allocate)     := homeFor(nextAddress)
      seen(allocate)      := 0.U
      sendBeat(allocate)  := 0.U
    }
    payload(allocate)(beatIndex(collectBeat)) := io.write.bits.data
    masks(allocate)(beatIndex(collectBeat)) := io.write.bits.mask
    slotState(allocate)                     := Mux(lastBeat, requestReady, filling)
    collectBeat                             := Mux(lastBeat, 0.U, collectBeat + 1.U)
    when(lastBeat) {
      nextAddress    := nextAddress + 64.U
      allocate       := nextSlot(allocate)
      allocatedLines := allocatedLines + 1.U
    }
  }

  val requestArb   = Module(new RRArbiter(new RequestFlit(p), slots))
  val requestQueue = Module(new Queue(new RequestFlit(p), 2, pipe = true))
  requestQueue.io.enq <> requestArb.io.out
  io.chi.req <> requestQueue.io.deq
  for (i <- 0 until slots) {
    val req = requestArb.io.in(i)
    req.valid                   := state === executing && slotState(i) === requestReady
    req.bits                    := 0.U.asTypeOf(new RequestFlit(p))
    req.bits.srcId              := nodeId.U
    req.bits.tgtId              := homes(i)
    req.bits.txnId              := i.U
    req.bits.returnNid          := nodeId.U
    req.bits.returnTxnId        := i.U
    req.bits.size               := 6.U
    req.bits.addr               := addresses(i)
    req.bits.opcode             := Mux(
      acquire,
      Opcode.CleanInvalid.U,
      Mux(reading, Opcode.ReadNoSnp.U, Opcode.WriteNoSnpPtl.U)
    )
    req.bits.allowRetry         := 1.U
    req.bits.snpAttr            := acquire.asUInt
    when(req.fire)(slotState(i) := waiting)
  }
  when(io.chi.req.fire) {
    val i = slotFor(io.chi.req.bits.txnId)
    assert(!inFlight(i), "NPU reused an in-flight CHI TxnID")
    inFlight(i) := true.B
  }

  // DBID is local to the responding Home, while TxnID identifies our ring slot.
  // The DAT arbiter may interleave lines and the queue holds each flit stable
  // across downstream backpressure.
  val dataArb   = Module(new RRArbiter(new DataFlit(p), slots))
  val dataQueue = Module(new Queue(new DataFlit(p), 2, pipe = true))
  dataQueue.io.enq <> dataArb.io.out
  io.chi.txDat <> dataQueue.io.deq
  for (i <- 0 until slots) {
    val dat = dataArb.io.in(i)
    dat.valid       := state === executing && slotState(i) === sending
    dat.bits        := 0.U.asTypeOf(new DataFlit(p))
    dat.bits.srcId  := nodeId.U
    dat.bits.tgtId  := homes(i)
    dat.bits.txnId  := dbids(i)
    dat.bits.opcode := Opcode.NonCopyBackWriteData.U
    dat.bits.dataId := sendBeat(i) * (p.dataBits / 128).U
    dat.bits.data   := payload(i)(beatIndex(sendBeat(i)))
    dat.bits.be     := masks(i)(beatIndex(sendBeat(i)))
    when(dat.fire) {
      sendBeat(i)                                               := sendBeat(i) + 1.U
      when(sendBeat(i) === (p.beatsPerLine - 1).U)(slotState(i) := waitComp)
    }
  }

  val rspError = WireDefault(false.B)
  val datError = WireDefault(false.B)
  io.chi.rxRsp.ready := state === executing
  when(io.chi.rxRsp.fire) {
    val r = io.chi.rxRsp.bits
    val i = slotFor(r.txnId)
    assert(
      r.txnId < slots.U && r.tgtId === nodeId.U && r.srcId === homes(i),
      "Invalid NPU response source or transaction"
    )
    rspError := r.respErr =/= 0.U
    when(writing && slotState(i) === waiting) {
      assert(r.opcode === Opcode.DBIDResp.U, "NPU write requires DBIDResp")
      dbids(i)     := r.dbid
      sendBeat(i)  := 0.U
      slotState(i) := sending
    }.otherwise {
      assert(
        r.opcode === Opcode.Comp.U &&
          ((acquire && slotState(i) === waiting) || (writing && slotState(i) === waitComp)),
        "Unexpected NPU completion"
      )
      when(acquire)(assert(r.respErr === 0.U, "Failed cache invalidation during NPU acquisition"))
      slotState(i) := complete
      inFlight(i)  := false.B
    }
  }

  io.chi.rxDat.ready              := state === executing && reading
  when(io.chi.rxDat.fire) {
    val d = io.chi.rxDat.bits
    val i = slotFor(d.txnId)
    val b = if (p.beatsPerLine == 1) 0.U(0.W) else d.dataId >> log2Ceil(p.dataBits / 128)
    assert(
      d.txnId < slots.U && d.tgtId === nodeId.U && d.srcId === homes(i) &&
        d.opcode === Opcode.CompData.U && slotState(i) === waiting,
      "Invalid NPU read response source or transaction"
    )
    assert(
      VecInit((0 until p.beatsPerLine).map(j => d.dataId === (j * p.dataBits / 128).U)).asUInt.orR,
      "Invalid NPU DataID"
    )
    assert(!(seen(i) & UIntToOH(b, p.beatsPerLine)).orR, "Duplicate NPU DataID")
    payload(i)(b) := d.data
    val nextSeen = seen(i) | UIntToOH(b, p.beatsPerLine)
    seen(i)  := nextSeen
    datError := d.respErr =/= 0.U
    when(nextSeen.andR) {
      slotState(i) := complete
      inFlight(i)  := false.B
    }
  }
  when(state === executing)(error := error || rspError || datError)

  io.read.valid               := state === executing && reading && slotState(retire) === complete
  io.read.bits.data           := payload(retire)(beatIndex(readBeat))
  io.read.bits.mask           := Fill(p.bytesPerBeat, 1.U(1.W))
  io.read.bits.last           := retiredLines === command.lines - 1.U && readBeat === (p.beatsPerLine - 1).U
  when(io.read.fire)(readBeat := Mux(readBeat === (p.beatsPerLine - 1).U, 0.U, readBeat + 1.U))
  val retireLine = state === executing && slotState(retire) === complete &&
    (!reading || (io.read.fire && readBeat === (p.beatsPerLine - 1).U))
  when(retireLine) {
    slotState(retire) := free
    retire            := nextSlot(retire)
    retiredLines      := retiredLines + 1.U
    when(retiredLines === command.lines - 1.U) {
      when(acquire) {
        io.publish := true.B
        leased     := true.B
      }
      state := respond
    }
  }

  io.chi.txRsp.valid       := false.B
  io.chi.txRsp.bits        := 0.U.asTypeOf(new ResponseFlit(p))
  io.chi.snp.ready         := true.B
  when(io.chi.snp.valid)(assert(false.B, "NPU region agent must never be a line-directory sharer"))
  io.done.valid            := state === respond
  io.done.bits             := error
  when(io.done.fire)(state := idle)
}
