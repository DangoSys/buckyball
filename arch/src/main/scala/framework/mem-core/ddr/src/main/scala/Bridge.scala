package memcore.memory.ddr

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.axi4
import memcore.bus.chi.snf.{LineRequest, LineResponse}

/** Joint-reset transport: AXI slave and bridge must discard old transactions together. */
@instantiable
class Bridge(p: Params) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Vec(p.clients, Flipped(Decoupled(new LineRequest(p.line))))
    val response = Vec(p.clients, Decoupled(new LineResponse(p.line)))
    val axi      = new axi4.Port(p.axi)
  })

  private val indexBits = log2Ceil(p.slots)
  private val beatBits  = math.max(1, log2Ceil(p.beats))
  val live              = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val write             = Reg(Vec(p.slots, Bool()))
  val tags              = Reg(Vec(p.slots, UInt(p.line.txnIdBits.W)))
  val addresses         = Reg(Vec(p.slots, UInt(p.line.addressBits.W)))
  val data              = Reg(Vec(p.slots, UInt(512.W)))
  val masks             = Reg(Vec(p.slots, UInt(64.W)))
  val arSent            = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val awSent            = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val wSent             = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val complete          = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val errors            = RegInit(VecInit(Seq.fill(p.slots)(false.B)))
  val readBeat          = RegInit(VecInit(Seq.fill(p.slots)(0.U(beatBits.W))))
  val awQueue           = Module(new Queue(UInt(indexBits.W), p.slots))
  val wQueue            = Module(new Queue(UInt(indexBits.W), p.slots))

  val arb = Module(new RRArbiter(UInt(indexBits.W), p.clients) {

    override lazy val lastGrant = {
      val last = RegInit(0.U(math.max(1, log2Ceil(p.clients)).W))
      when(io.out.fire)(last := io.chosen)
      last
    }

  })

  for (c <- 0 until p.clients) {
    val indices   = c * p.slotsPerClient until (c + 1) * p.slotsPerClient
    val free      = VecInit(indices.map(i => !live(i)))
    // Tags belong to each client. A reused live tag waits for its old response.
    val duplicate = indices.map(i => live(i) && tags(i) === io.request(c).bits.id).reduce(_ || _)
    val eligible  = free.asUInt.orR && !duplicate &&
      (!io.request(c).bits.write || (awQueue.io.enq.ready && wQueue.io.enq.ready)) && !reset.asBool
    arb.io.in(c).valid  := io.request(c).valid && eligible
    arb.io.in(c).bits   := (c * p.slotsPerClient).U(indexBits.W) + PriorityEncoder(free.asUInt)
    io.request(c).ready := arb.io.in(c).ready && eligible
  }
  arb.io.out.ready := true.B
  val allocated = arb.io.out.bits
  val request   = io.request(arb.io.chosen).bits
  awQueue.io.enq.valid := arb.io.out.fire && request.write
  wQueue.io.enq.valid  := awQueue.io.enq.valid
  awQueue.io.enq.bits  := allocated
  wQueue.io.enq.bits   := allocated
  when(arb.io.out.fire) {
    assert(request.addr(5, 0) === 0.U, "DDR line request must be 64 byte aligned")
    live(allocated)     := true.B; write(allocated)                                := request.write
    tags(allocated)     := request.id; addresses(allocated)                        := request.addr
    data(allocated)     := Mux(request.write, request.data, 0.U); masks(allocated) := request.mask
    arSent(allocated)   := false.B; awSent(allocated)                              := false.B; wSent(allocated)    := false.B
    complete(allocated) := false.B; errors(allocated)                              := false.B; readBeat(allocated) := 0.U
  }

  def address(slot: UInt): axi4.Address = {
    val a = Wire(new axi4.Address(p.axi))
    a     := 0.U.asTypeOf(a)
    a.id  := slot +& p.idBase.U; a.addr := addresses(slot)
    a.len := (p.beats - 1).U; a.size    := log2Ceil(p.axi.bytes).U; a.burst := 1.U
    a
  }

  def select(candidates: Vec[Bool], cursor: UInt): UInt = {
    val count = candidates.length
    MuxCase(
      0.U(math.max(1, log2Ceil(count)).W),
      (0 until count).map { off =>
        val sum   = cursor +& off.U
        val index = Mux(sum >= count.U, sum - count.U, sum)(math.max(1, log2Ceil(count)) - 1, 0)
        candidates(index) -> index
      }
    )
  }

  val arCursor = RegInit(0.U(indexBits.W))
  val arOffer  = RegInit(false.B)
  val arIndex  = Reg(UInt(indexBits.W))
  val reads    = VecInit((0 until p.slots).map(i => live(i) && !write(i) && !arSent(i)))
  when(!arOffer && reads.asUInt.orR) { arOffer := true.B; arIndex := select(reads, arCursor) }
  io.axi.ar.valid := arOffer && !reset.asBool
  io.axi.ar.bits                                   := address(arIndex)
  when(io.axi.ar.fire) {
    arOffer := false.B; arSent(arIndex) := true.B; arCursor := Mux(arIndex === (p.slots - 1).U, 0.U, arIndex + 1.U)
  }
  io.axi.aw.valid                                  := awQueue.io.deq.valid && !reset.asBool
  io.axi.aw.bits                                   := address(awQueue.io.deq.bits)
  awQueue.io.deq.ready                             := io.axi.aw.ready && !reset.asBool
  when(io.axi.aw.fire)(awSent(awQueue.io.deq.bits) := true.B)
  val wBeat = RegInit(0.U(beatBits.W))
  val wLast = wBeat === (p.beats - 1).U
  io.axi.w.valid      := wQueue.io.deq.valid && !reset.asBool
  io.axi.w.bits.data  := (data(wQueue.io.deq.bits) >> (wBeat * p.dataBits.U))(p.dataBits - 1, 0)
  io.axi.w.bits.strb  := (masks(wQueue.io.deq.bits) >> (wBeat * p.axi.bytes.U))(p.axi.bytes - 1, 0)
  io.axi.w.bits.last  := wLast
  wQueue.io.deq.ready := io.axi.w.ready && wLast && !reset.asBool
  when(io.axi.w.fire) {
    when(wLast) { wBeat := 0.U; wSent(wQueue.io.deq.bits) := true.B }
      .otherwise(wBeat := wBeat + 1.U)
  }
  val rIndex = (io.axi.r.bits.id - p.idBase.U)(indexBits - 1, 0)
  val bIndex = (io.axi.b.bits.id - p.idBase.U)(indexBits - 1, 0)
  val rKnown = io.axi.r.bits.id >= p.idBase.U && io.axi.r.bits.id < (p.idBase + p.slots).U &&
    live(rIndex) && !write(rIndex) && arSent(rIndex) && !complete(rIndex)
  val bKnown = io.axi.b.bits.id >= p.idBase.U && io.axi.b.bits.id < (p.idBase + p.slots).U &&
    live(bIndex) && write(bIndex) && awSent(bIndex) && wSent(bIndex) && !complete(bIndex)
  io.axi.r.ready := rKnown && !reset.asBool
  io.axi.b.ready := bKnown && !reset.asBool
  when(io.axi.r.valid && !reset.asBool)(assert(rKnown, "DDR received R for an inactive AXI ID"))
  when(io.axi.b.valid && !reset.asBool)(assert(bKnown, "DDR received B before AW and complete W or for inactive ID"))
  when(io.axi.r.fire) {
    assert(io.axi.r.bits.resp =/= 1.U, "DDR nonexclusive read received EXOKAY")
    val last  = readBeat(rIndex) === (p.beats - 1).U
    assert(io.axi.r.bits.last === last, "DDR AXI RLAST does not match line beat count")
    val shift = readBeat(rIndex) * p.dataBits.U
    data(rIndex)                  := data(rIndex) | (io.axi.r.bits.data.pad(512) << shift)
    errors(rIndex)                := errors(rIndex) || io.axi.r.bits.resp(1)
    when(last && io.axi.r.bits.last)(complete(rIndex) := true.B)
      .otherwise(readBeat(rIndex) := readBeat(rIndex) + 1.U)
  }
  when(io.axi.b.fire) {
    assert(io.axi.b.bits.resp =/= 1.U, "DDR nonexclusive write received EXOKAY")
    complete(bIndex) := true.B; errors(bIndex) := io.axi.b.bits.resp(1); data(bIndex) := 0.U
  }
  for (c <- 0 until p.clients) {
    val cursor = RegInit(0.U(math.max(1, log2Ceil(p.slotsPerClient)).W))
    val offer  = RegInit(false.B)
    val index  = Reg(UInt(indexBits.W))
    val ready  = VecInit((c * p.slotsPerClient until (c + 1) * p.slotsPerClient).map(i => live(i) && complete(i)))
    when(!offer && ready.asUInt.orR) {
      offer := true.B; index := (c * p.slotsPerClient).U(indexBits.W) + select(ready, cursor)
    }
    io.response(c).valid := offer && !reset.asBool
    io.response(c).bits.id    := tags(index)
    io.response(c).bits.error := errors(index)
    io.response(c).bits.data  := Mux(errors(index), 0.U, data(index))
    when(io.response(c).fire) {
      offer  := false.B; live(index) := false.B
      cursor := Mux(index === ((c + 1) * p.slotsPerClient - 1).U, 0.U, index - (c * p.slotsPerClient).U + 1.U)
    }
  }
}
