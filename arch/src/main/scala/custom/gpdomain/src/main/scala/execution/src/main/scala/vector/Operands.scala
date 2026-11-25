package framework.custom.gpdomain.execution.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.acc.gpdomain.configs.GpDomainParam

/**
 * The operand words of a batch, read from and written to the register file. A register has one
 * word in every bank, and the elements of a batch are consecutive, so a kind of operand (the
 * mask, operand, source, destination or accumulator of a read; the result of a write) has at most
 * one word in a bank. Each bank keeps the word it last read for each kind: a read fetches, a bank
 * a cycle, the words its kinds do not hold, and is answered the cycle the last one is in. A write
 * is one word a bank, merged from the lanes that share it. A permute's source comes from a
 * snapshot of its register group, a word a cycle.
 *
 * The words of a kind follow its first port's in order: a port's word is a few words on, by the
 * element width and where the batch starts in a word (`steps`). So the banks' words are turned
 * until the first port's is first, and a port chooses among its few; a write's words are gathered
 * the same way from the origin of its batch and turned back. No port reaches every bank.
 */
@instantiable
class Operands(val gpParam: GpDomainParam) extends Module {
  val p                = gpParam.execution
  val wordsPerRegister = p.wordsPerRegister
  val bankCount        = p.registerBanks
  val bankBits         = log2Ceil(bankCount)
  val addressBits      = log2Ceil(32 * wordsPerRegister)
  val highBits         = addressBits - bankBits
  val count            = math.max(p.laneNumber, p.memoryPorts)
  // A read is every operand of a batch: the mask, operand, source, destination and accumulator words, `count` each.
  val phases           = 5
  val reads            = phases * count
  val snapshotWords    = 8 * wordsPerRegister
  val snapshotBits     = log2Ceil(snapshotWords)
  require(bankBits > 0)
  require(p.laneNumber % p.memoryPorts == 0)

  // The element a port takes, counted from the first port's: the next one, a bank row a memory port, or one a
  // memory port (VectorThread.portElement).
  val rowSlots    = count / p.memoryPorts
  val memoryLanes = math.min(p.laneNumber, p.memoryPorts)
  def offsets(port:   Int): Set[Int] = Set(port, port % memoryLanes) ++
    (0 to p.maxSew).map(width => (port / rowSlots) * math.min(16 >> width, rowSlots) + port % rowSlots)

  // The words a port's mask bit can be past the first port's, by the bit the batch starts at.
  def maskSteps(port: Int): Seq[Int] =
    offsets(port).flatMap(offset =>
      (0 until p.wordBits).map(start => (start + offset) >> p.wordOffsetBits)
    ).toSeq.sorted

  // The words a port's element or mask bit can be past the first port's: elements are 2^shift a word.
  def steps(port: Int): Seq[Int] = (maskSteps(port) ++ offsets(port).flatMap { offset =>
    (0 to p.maxSew).flatMap(shift => (0 until (1 << shift)).map(start => (start + offset) >> shift))
  }).distinct.sorted

  // The words of a kind a port can read: the mask is bits, the accumulator the first port's word.
  def readSteps(phase: Int, port: Int): Seq[Int] =
    if (phase == 0) maskSteps(port) else if (phase == 4) Seq(0) else steps(port)

  // Turned words: word `index` is the bank's `index + by` on.
  def turn[T <: Data](words: Seq[T], by: UInt): Seq[T] = (0 until bankBits).foldLeft(words) { (stage, bit) =>
    stage.indices.map(index => Mux(by(bit), stage((index + (1 << bit)) % bankCount), stage(index)))
  }

  @public
  val io = IO(new Bundle {
    val initialize = Input(Bool())
    val busy       = Output(Bool())

    val read = Flipped(Decoupled(Vec(
      reads,
      new Bundle {
        val valid    = Bool()
        val address  = UInt(addressBits.W)
        val snapshot = Bool()
      }
    )))

    val readResult = Decoupled(Vec(reads, UInt(p.wordBits.W)))

    // `origin` is the word of the batch's first element, written or not.
    val write = Flipped(Decoupled(new Bundle {
      val origin = UInt(addressBits.W)

      val lanes = Vec(
        p.laneNumber,
        new Bundle {
          val valid   = Bool()
          val address = UInt(addressBits.W)
          val data    = UInt(p.wordBits.W)
          val mask    = UInt(p.wordBits.W)
        }
      )

    }))

    val writeDone    = Decoupled(Bool())
    val snapshot     = Flipped(Decoupled(UInt(5.W)))
    val snapshotDone = Decoupled(Bool())
  })

  val vrf = Instantiate(new VRF(gpParam))
  vrf.io.initialize := io.initialize
  val idle :: reading :: writing :: copying :: copyDone :: Nil = Enum(5)
  val state                                                    = RegInit(idle)
  val readFrame                                                = Reg(chiselTypeOf(io.read.bits))
  val writeFrame                                               = Reg(chiselTypeOf(io.write.bits))

  // The word each bank last read for each kind of operand.
  val keptValid = RegInit(VecInit(Seq.fill(bankCount)(VecInit(Seq.fill(phases)(false.B)))))
  val keptHigh  = Reg(Vec(bankCount, Vec(phases, UInt(highBits.W))))
  val keptData  = Reg(Vec(bankCount, Vec(phases, UInt(p.wordBits.W))))
  // The kind whose word a bank's response carries, and the banks a write has reached.
  val fetching  = RegInit(VecInit(Seq.fill(bankCount)(false.B)))
  val fetched   = Reg(Vec(bankCount, UInt(log2Ceil(phases).W)))
  val written   = Reg(Vec(bankCount, Bool()))
  val settled   = Reg(Vec(bankCount, Bool()))

  val scratch        = SyncReadMem(snapshotWords, UInt(p.wordBits.W))
  val scratchEnable  = WireDefault(false.B)
  val scratchWrite   = WireDefault(false.B)
  val scratchAddress = WireDefault(0.U(snapshotBits.W))
  val scratchData    = WireDefault(0.U(p.wordBits.W))
  val scratchRead    = scratch.readWrite(scratchAddress, scratchData, scratchEnable && !io.initialize, scratchWrite)
  val scratchPending = RegInit(false.B)
  val scratchLane    = Reg(UInt(log2Ceil(count).W))
  val scratchDone    = Reg(Vec(count, Bool()))
  val scratchWords   = Reg(Vec(count, UInt(p.wordBits.W)))
  val snapshotValid  = RegInit(false.B)
  val source         = Reg(UInt(5.W))
  val copyIndex      = Reg(UInt(snapshotBits.W))
  val copyPending    = RegInit(false.B)

  val available = state === idle && !vrf.io.busy && !io.initialize
  io.read.ready         := available
  io.write.ready        := available
  io.snapshot.ready     := available
  io.writeDone.bits     := true.B
  io.snapshotDone.valid := state === copyDone && !io.initialize
  io.snapshotDone.bits  := true.B
  io.busy               := state =/= idle || vrf.io.busy || io.initialize

  def slot(phase: Int, lane: Int) = readFrame(phase * count + lane)
  val copyAddress = (source * wordsPerRegister.U + copyIndex)(addressBits - 1, 0)

  // The words of the read each bank holds, by kind.
  val words   = Wire(Vec(phases, Vec(bankCount, UInt(p.wordBits.W))))
  val holding = Wire(Vec(bankCount, Bool()))
  // A kind's words start at its first port's; the last port's is the furthest on.
  def bankOf(address: UInt): UInt = address(bankBits - 1, 0)
  def highOf(address: UInt): UInt = address(addressBits - 1, bankBits)
  val firsts = (0 until phases).map(phase => slot(phase, 0).address)
  val lasts  = (0 until phases).map(phase => slot(phase, count - 1).address)

  // A write's words by step from its origin, each from the lanes that can be there, then turned to their banks.
  val writeOrigin = bankOf(writeFrame.origin)
  val writeSteps  = writeFrame.lanes.map(w => bankOf(w.address) - writeOrigin)

  val stepWrites = (0 until bankCount).map { step =>
    val sources = (0 until p.laneNumber).filter(lane => steps(lane).contains(step)).map { lane =>
      (writeFrame.lanes(lane), writeFrame.lanes(lane).valid && writeSteps(lane) === step.U)
    }
    val word    = Wire(new Bundle {
      val valid = Bool()
      val mask  = UInt(p.wordBits.W)
      val data  = UInt(p.wordBits.W)
    })
    word.valid := sources.map(_._2).reduce(_ || _)
    word.mask := sources.map { case (w, here) => Mux(here, w.mask, 0.U) }.reduce(_ | _)
    word.data := sources.map { case (w, here) => Mux(here, w.data & w.mask, 0.U) }.reduce(_ | _)
    word
  }

  val bankWrites = turn(stepWrites, 0.U - writeOrigin)

  // The banks before the origin's hold the words that ran into the next register.
  val wrappedHigh = PriorityMux(
    writeFrame.lanes.map(w => w.valid && bankOf(w.address) < writeOrigin),
    writeFrame.lanes.map(w => highOf(w.address))
  )

  for (bank <- 0 until bankCount) {
    val request  = vrf.io.request(bank)
    val response = vrf.io.response(bank)
    request.valid        := false.B
    request.bits.address := 0.U
    request.bits.write   := false.B
    request.bits.data    := 0.U
    request.bits.mask    := 0.U
    response.ready       := true.B

    // The word a kind needs here: the one of its lanes whose address is in this bank.
    val lanes = (0 until phases).map { phase =>
      VecInit((0 until count).map { lane =>
        val entry = slot(phase, lane)
        entry.valid && !entry.snapshot && entry.address(bankBits - 1, 0) === bank.U
      })
    }
    val wants = VecInit(lanes.map(_.asUInt.orR))
    val highs = VecInit((0 until phases).map { phase =>
      Mux(bank.U < bankOf(firsts(phase)), highOf(lasts(phase)), highOf(firsts(phase)))
    })
    val hits  = (0 until phases).map { phase =>
      VecInit((0 until phases).map(kept => keptValid(bank)(kept) && keptHigh(bank)(kept) === highs(phase)))
    }
    for (phase <- 0 until phases) words(phase)(bank) := Mux1H(hits(phase), keptData(bank))
    val missing = VecInit((0 until phases).map { phase =>
      wants(phase) && !hits(phase).asUInt.orR && !(fetching(bank) && fetched(bank) === phase.U)
    })
    holding(bank) := !(0 until phases).map(phase => wants(phase) && !hits(phase).asUInt.orR).reduce(_ || _)

    when(response.fire && fetching(bank)) {
      fetching(bank)                 := false.B
      keptValid(bank)(fetched(bank)) := true.B
      keptData(bank)(fetched(bank))  := response.bits
    }
    when(state === reading && !io.initialize) {
      for {
        phase <- 0 until phases
        lane  <- 0 until count
      } when(lanes(phase)(lane)) {
        assert(
          slot(phase, lane).address(addressBits - 1, bankBits) === highs(phase),
          "The lanes of a kind read one word of a bank"
        )
      }
      val phase = PriorityEncoder(missing)
      request.valid        := missing.asUInt.orR
      request.bits.address := Cat(highs(phase), bank.U(bankBits.W))
      when(request.fire) {
        fetching(bank)         := true.B
        fetched(bank)          := phase
        keptValid(bank)(phase) := false.B
        keptHigh(bank)(phase)  := highs(phase)
      }
    }

    // A write: the lanes of this bank share one word.
    val writers = VecInit(writeFrame.lanes.map(w => w.valid && bankOf(w.address) === bank.U))
    val high    = Mux(bank.U < writeOrigin, wrappedHigh, highOf(writeFrame.origin))
    val mask    = bankWrites(bank).mask
    val data    = bankWrites(bank).data
    when(state === writing && !io.initialize) {
      assert(bankWrites(bank).valid === writers.asUInt.orR, "A lane's word is a step it can take from the origin")
      assert(
        mask === writeFrame.lanes.zip(writers).map { case (w, mine) => Mux(mine, w.mask, 0.U) }.reduce(_ | _) &&
          data === writeFrame.lanes.zip(writers).map { case (w, mine) => Mux(mine, w.data & w.mask, 0.U) }.reduce(_ | _),
        "A bank's word is merged from its lanes"
      )
      request.valid                                                                 := bankWrites(bank).valid && !written(bank)
      request.bits.address                                                          := Cat(high, bank.U(bankBits.W))
      request.bits.write                                                            := true.B
      request.bits.mask                                                             := mask
      request.bits.data                                                             := data
      when(request.fire) {
        for ((w, mine) <- writeFrame.lanes.zip(writers))
          when(mine)(assert(highOf(w.address) === high, "The lanes of a bank write one word"))
        written(bank) := true.B
        for (kept <- 0 until phases)
          when(keptValid(bank)(kept) && keptHigh(bank)(kept) === high) {
            keptData(bank)(kept) := (keptData(bank)(kept) & ~mask) | data
          }
      }
      when(!bankWrites(bank).valid || written(bank) && response.fire)(settled(bank) := true.B)
    }

    when(state === copying && copyAddress(bankBits - 1, 0) === bank.U && !io.initialize) {
      request.valid                  := !copyPending
      request.bits.address           := copyAddress
      when(request.fire)(copyPending := true.B)
      when(response.fire && copyPending) {
        scratchEnable         := true.B
        scratchWrite          := true.B
        scratchAddress        := copyIndex
        scratchData           := response.bits
        copyPending           := false.B
        when(copyIndex === (snapshotWords - 1).U) {
          snapshotValid := true.B
          state         := copyDone
        }.otherwise(copyIndex := copyIndex + 1.U)
      }
    }
  }

  // A permute's source words, a lane a cycle from the snapshot.
  val scratchWanted =
    VecInit((0 until count).map(lane => slot(2, lane).valid && slot(2, lane).snapshot && !scratchDone(lane)))
  when(state === reading && !io.initialize) {
    when(scratchWanted.asUInt.orR && !scratchPending) {
      val lane = PriorityEncoder(scratchWanted)
      scratchEnable  := true.B
      scratchAddress := VecInit((0 until count).map(l => slot(2, l).address(snapshotBits - 1, 0)))(lane)
      scratchPending := true.B
      scratchLane    := lane
    }
    when(scratchPending) {
      scratchPending            := false.B
      scratchWords(scratchLane) := scratchRead
      scratchDone(scratchLane)  := true.B
    }
  }

  io.readResult.valid := state === reading && holding.asUInt.andR && !scratchWanted.asUInt.orR && !scratchPending &&
    !io.initialize
  for (phase <- 0 until phases) {
    val turned = turn(words(phase), bankOf(firsts(phase)))
    for (lane <- 0 until count) {
      val entry = slot(phase, lane)
      val step  = bankOf(entry.address) - bankOf(firsts(phase))
      val word  = Mux1H(readSteps(phase, lane).map(on => (step === on.U) -> turned(on)))
      io.readResult.bits(phase * count + lane) := (if (phase == 2) Mux(entry.snapshot, scratchWords(lane), word)
                                                   else word)
      when(state === reading && entry.valid && !entry.snapshot && !io.initialize) {
        assert(word === words(phase)(bankOf(entry.address)), "A port's word is a step it can take from the first port's")
      }
    }
  }
  io.writeDone.valid := state === writing && settled.asUInt.andR && !io.initialize

  when(available) {
    assert(PopCount(VecInit(Seq(io.read.valid, io.write.valid, io.snapshot.valid))) <= 1.U)
  }
  when(io.read.fire) {
    assert(snapshotValid || !io.read.bits.map(r => r.valid && r.snapshot).reduce(_ || _))
    readFrame   := io.read.bits
    scratchDone := VecInit(Seq.fill(count)(false.B))
    state       := reading
  }
  when(io.write.fire) {
    writeFrame := io.write.bits
    written    := VecInit(Seq.fill(bankCount)(false.B))
    settled    := VecInit(Seq.fill(bankCount)(false.B))
    state      := writing
  }
  when(io.snapshot.fire) {
    source        := io.snapshot.bits
    copyIndex     := 0.U
    snapshotValid := false.B
    state         := copying
  }
  when(io.readResult.fire || io.writeDone.fire || io.snapshotDone.fire)(state := idle)
  when(io.initialize) {
    state          := idle
    fetching       := VecInit(Seq.fill(bankCount)(false.B))
    copyPending    := false.B
    scratchPending := false.B
    snapshotValid  := false.B
    keptValid.foreach(_.foreach(_ := false.B))
  }
}
