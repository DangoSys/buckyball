package memcore.memory.uncached_ram

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.chi.snf.{LineRequest, LineResponse}
import memcore.bus.chi.rnf.CacheAtomic

/** Central normal-RAM ordering point. External DMA participates through control ranges, with an independent data path. */
@instantiable
class Ram(p: Params) extends Module {
  private val slotBits   = log2Ceil(p.slots)
  private val sourceBits = math.max(1, log2Ceil(p.sources))
  private val tagBits    = math.max(p.tagBits, p.line.txnIdBits)

  @public
  val io = IO(new Bundle {
    val cpuRequest     = Vec(p.cpus, Flipped(Decoupled(new Request(p))))
    val cpuResponse    = Vec(p.cpus, Decoupled(new Response(p)))
    val lineRequest    = Vec(p.lineAgents, Flipped(Decoupled(new LineRequest(p.line))))
    val lineResponse   = Vec(p.lineAgents, Decoupled(new LineResponse(p.line)))
    val memoryRequest  = Vec(p.ports, Decoupled(new LineRequest(p.line)))
    val memoryResponse = Vec(p.ports, Flipped(Decoupled(new LineResponse(p.line))))
    val external       = Input(Vec(p.externalPorts, new ExternalAccess(p)))
    val externalAllow  = Output(Vec(p.externalPorts, Bool()))
    val outstanding    = Output(UInt(log2Ceil(p.slots + 1).W))
  })

  class Command extends Bundle {
    val source = UInt(sourceBits.W); val tag = UInt(tagBits.W)
    val addr   = UInt(64.W); val size        = UInt(3.W); val write    = Bool(); val atomic = UInt(4.W)
    val data   = UInt(512.W); val mask       = UInt(64.W); val operand = UInt(64.W)
  }

  val free :: readSend :: readWait :: writeSend :: writeWait :: complete :: Nil = Enum(6)
  val state                                                                     = RegInit(VecInit(Seq.fill(p.slots)(free)))
  val command                                                                   = Reg(Vec(p.slots, new Command))
  val answer                                                                    = Reg(Vec(p.slots, UInt(512.W)))
  val error                                                                     = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val writeData                                                                 = Reg(Vec(p.slots, UInt(512.W)))
  val reservation                                                               = RegInit(VecInit(Seq.fill(p.cpus)(false.B)))
  val reservedAddr                                                              = Reg(Vec(p.cpus, UInt(64.W)))
  val reservedSize                                                              = Reg(Vec(p.cpus, UInt(3.W)))
  val available                                                                 = VecInit(state.map(_ === free)).asUInt
  val allocated                                                                 = PriorityEncoder(available)
  io.outstanding := PopCount(state.map(_ =/= free))

  def overlap(addr: UInt, range: ExternalAccess): Bool = {
    val end      = range.addr.pad(p.line.addressBits + 1) + range.bytes
    val lineBase = Cat(addr(p.line.addressBits - 1, 6), 0.U(6.W)).pad(p.line.addressBits + 1)
    range.valid && lineBase < end && range.addr < lineBase + 64.U
  }

  for (i <- 0 until p.externalPorts) {
    io.externalAllow(i) := !(0 until p.slots).map(slot =>
      state(slot) =/= free && state(slot) =/= complete && overlap(command(slot).addr, io.external(i))
    ).reduce(_ || _)
  }

  def rr[T <: Data](gen: T, n: Int): RRArbiter[T] = Module(new RRArbiter(gen, n) {

    override lazy val lastGrant = {
      val previous = RegInit(0.U(math.max(1, log2Ceil(n)).W))
      when(io.out.fire)(previous := io.chosen)
      previous
    }

  })

  val admit = rr(new Command, p.sources)
  admit.io.out.ready := available.orR && !reset.asBool
  for (source <- 0 until p.sources) {
    val packet = WireDefault(0.U.asTypeOf(new Command))
    val valid  = Wire(Bool()); val ready = Wire(Bool())
    packet.source := source.U
    if (source < p.cpus) {
      val request = io.cpuRequest(source)
      packet.tag   := request.bits.tag; packet.addr     := request.bits.addr; packet.size      := request.bits.size
      packet.write := request.bits.write; packet.atomic := request.bits.atomic; packet.operand := request.bits.data
      val mask = MuxLookup(request.bits.size, 255.U(64.W))(Seq(0.U -> 1.U(64.W), 1.U -> 3.U(64.W), 2.U -> 15.U(64.W)))
      packet.mask := (mask << request.bits.addr(5, 0))(63, 0)
      packet.data := (request.bits.data.pad(512) << (request.bits.addr(5, 0) << 3))(511, 0)
      valid       := request.valid; request.ready := ready
      when(request.valid) {
        assert(request.bits.size <= 3.U, "Uncached RAM invalid CPU size")
        assert(request.bits.atomic <= CacheAtomic.SC.U, "Uncached RAM invalid atomic")
        assert(
          request.bits.atomic === 0.U || (!request.bits.write && request.bits.size >= 2.U),
          "Uncached RAM atomic operand contract"
        )
      }
    } else {
      val request = io.lineRequest(source - p.cpus)
      packet.tag  := request.bits.id; packet.addr   := request.bits.addr
      packet.size := 6.U; packet.write              := request.bits.write
      packet.data := request.bits.data; packet.mask := request.bits.mask
      valid       := request.valid; request.ready   := ready
      when(request.valid)(assert(request.bits.addr(5, 0) === 0.U, "Uncached RAM line request is not aligned"))
    }
    val externalConflict =
      if (p.externalPorts == 0) false.B
      else
        io.external.map(range => overlap(packet.addr, range)).reduce(_ || _)
    val locked    = externalConflict || (0 until p.slots).map(i =>
      state(i) =/= free && state(i) =/= complete && command(i).addr(63, 6) === packet.addr(63, 6)
    ).reduce(_ || _)
    // Actual CpuMem has one outstanding request per CPU; line agents retain independent caller IDs.
    val duplicate = (0 until p.slots).map(i =>
      state(i) =/= free && command(i).source === source.U &&
        (if (source < p.cpus) true.B else command(i).tag === packet.tag)
    ).reduce(_ || _)
    admit.io.in(source).valid := valid && !locked && !duplicate && !reset.asBool
    admit.io.in(source).bits  := packet
    ready                     := admit.io.in(source).ready && !locked && !duplicate && !reset.asBool
  }
  when(admit.io.out.fire) {
    val request   = admit.io.out.bits
    val cpu       = request.source < p.cpus.U
    val cpuIndex  = request.source(math.max(1, log2Ceil(p.cpus)) - 1, 0)
    val last      = request.addr.pad(65) + ((1.U(65.W) << request.size) - 1.U)
    val bad       = request.addr < p.base.U || last >= (p.base + p.bytes).U ||
      (request.addr & ((1.U(64.W) << request.size) - 1.U)).orR
    val sc        = cpu && request.atomic === CacheAtomic.SC.U
    val scSuccess =
      reservation(cpuIndex) && reservedAddr(cpuIndex) === request.addr && reservedSize(cpuIndex) === request.size
    val writing   =
      request.write || (cpu && request.atomic =/= CacheAtomic.None.U && request.atomic =/= CacheAtomic.LR.U && (!sc || scSuccess))
    command(allocated)                                                             := request; answer(allocated) := 0.U; error(allocated) := bad
    writeData(allocated)                                                           := request.data
    state(allocated)                                                               := Mux(bad || (sc && !scSuccess), complete, Mux(request.write || sc, writeSend, readSend))
    when(sc && !bad && !scSuccess)(answer(allocated)                               := 1.U)
    when(cpu && (request.atomic === CacheAtomic.LR.U || sc))(reservation(cpuIndex) := false.B)
    when(!bad && writing && request.mask.orR) {
      for (hart <- 0 until p.cpus) {
        when(reservation(hart) && reservedAddr(hart)(63, 6) === request.addr(63, 6))(reservation(hart) := false.B)
      }
    }
  }

  for (port <- 0 until p.ports) {
    val send = rr(UInt(slotBits.W), p.slotsPerPort)
    for (local <- 0 until p.slotsPerPort) {
      val slot = port * p.slotsPerPort + local
      send.io.in(local).valid := state(slot) === readSend || state(slot) === writeSend
      send.io.in(local).bits  := slot.U
    }
    val offer = RegInit(false.B)
    val offeredSlot = Reg(UInt(slotBits.W))
    val packet      = Reg(new LineRequest(p.line))
    send.io.out.ready            := !offer
    when(send.io.out.fire) {
      val slot = send.io.out.bits
      offer        := true.B; offeredSlot          := slot
      packet.id    := slot; packet.addr            := Cat(command(slot).addr(p.line.addressBits - 1, 6), 0.U(6.W))
      packet.write := state(slot) === writeSend
      packet.data  := writeData(slot); packet.mask := command(slot).mask
    }
    io.memoryRequest(port).valid := offer && !reset.asBool
    io.memoryRequest(port).bits  := packet
    when(io.memoryRequest(port).fire) {
      offer              := false.B
      state(offeredSlot) := Mux(packet.write, writeWait, readWait)
    }
    val response = io.memoryResponse(port)
    val inRange = response.bits.id >= (port * p.slotsPerPort).U && response.bits.id < ((port + 1) * p.slotsPerPort).U
    val slot    = response.bits.id(slotBits - 1, 0)
    response.ready := !reset.asBool
    when(response.fire) {
      assert(
        inRange && (state(slot) === readWait || state(slot) === writeWait),
        "Uncached RAM response has no in-flight owner"
      )
      when(inRange && (state(slot) === readWait || state(slot) === writeWait)) {
        val request     = command(slot)
        val cpu         = request.source < p.cpus.U
        val old         = (response.bits.data >> (request.addr(5, 0) << 3))(63, 0)
        val word        = request.size === 2.U
        val left        = Mux(word, Cat(0.U(32.W), old(31, 0)), old)
        val right       = Mux(word, Cat(0.U(32.W), request.operand(31, 0)), request.operand)
        val signedLeft  = Mux(word, Cat(Fill(32, old(31)), old(31, 0)), old).asSInt
        val signedRight = Mux(word, Cat(Fill(32, right(31)), right(31, 0)), right).asSInt
        val updated     = MuxLookup(request.atomic, right)(Seq(
          CacheAtomic.Add.U  -> (left + right),
          CacheAtomic.Xor.U  -> (left ^ right),
          CacheAtomic.And.U  -> (left & right),
          CacheAtomic.Or.U   -> (left | right),
          CacheAtomic.Min.U  -> Mux(signedLeft < signedRight, left, right),
          CacheAtomic.Max.U  -> Mux(signedLeft > signedRight, left, right),
          CacheAtomic.MinU.U -> Mux(left < right, left, right),
          CacheAtomic.MaxU.U -> Mux(left > right, left, right)
        ))
        error(slot) := response.bits.error
        when(response.bits.error) { answer(slot) := 0.U; state(slot) := complete }
          .elsewhen(state(slot) === writeWait)(state(slot) := complete)
          .otherwise {
            val atomic = request.atomic =/= CacheAtomic.None.U
            val mask   = MuxLookup(request.size, "hffffffffffffffff".U(64.W))(Seq(
              0.U -> 255.U(64.W),
              1.U -> 65535.U(64.W),
              2.U -> "hffffffff".U(64.W)
            ))
            answer(slot) := Mux(cpu, Mux(atomic, signedLeft.asUInt, old & mask), response.bits.data)
            when(cpu && request.atomic === CacheAtomic.LR.U) {
              val hart = request.source(math.max(1, log2Ceil(p.cpus)) - 1, 0)
              reservation(hart) := true.B; reservedAddr(hart) := request.addr; reservedSize(hart) := request.size
            }
            val rmw    = cpu && atomic && request.atomic =/= CacheAtomic.LR.U
            when(rmw) {
              writeData(slot) := (updated.pad(512) << (request.addr(5, 0) << 3))(511, 0)
              state(slot)     := writeSend
            }.otherwise(state(slot) := complete)
          }
      }
    }
  }
  // Invalidating after LR response handling prevents a same-cycle stale reservation.
  for {
    range   <- io.external
    hart    <- 0 until p.cpus
  } {
    when(range.write && overlap(reservedAddr(hart), range))(reservation(hart) := false.B)
  }

  for (source <- 0 until p.sources) {
    val offer      = RegInit(false.B)
    val selected   = Reg(UInt(slotBits.W))
    val candidates = VecInit((0 until p.slots).map(i => state(i) === complete && command(i).source === source.U))
    when(!offer && candidates.asUInt.orR) { offer := true.B; selected := PriorityEncoder(candidates) }
    val fire       = Wire(Bool())
    if (source < p.cpus) {
      val response = io.cpuResponse(source)
      response.valid      := offer && !reset.asBool
      response.bits.tag   := command(selected).tag; response.bits.data := answer(selected)(63, 0);
      response.bits.error := error(selected)
      fire                := response.fire
    } else {
      val response = io.lineResponse(source - p.cpus)
      response.valid      := offer && !reset.asBool
      response.bits.id    := command(selected).tag; response.bits.data := answer(selected);
      response.bits.error := error(selected)
      fire                := response.fire
    }
    when(fire) { offer := false.B; state(selected) := free }
  }
}
