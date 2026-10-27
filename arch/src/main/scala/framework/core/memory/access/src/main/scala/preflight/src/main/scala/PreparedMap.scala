package framework.core.memory.access.preflight

import framework.core.memory.access.preflight.{Params => ModuleParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class MapTag(b: ModuleParam) extends Bundle {
  val p  = b
  val id = UInt(p.idBits.W)
}

class MapReady(b: ModuleParam) extends Bundle {
  val p = b

  val id    = UInt(p.idBits.W)
  val error = UInt(3.W)
  val va    = UInt(64.W)
}

class MapQuery(b: ModuleParam) extends Bundle {
  val p = b

  val valid = Bool()
  val id    = UInt(p.idBits.W)
  val va    = UInt(64.W)
  val bytes = UInt(13.W)
  val write = Bool()
}

class MapResult(b: ModuleParam) extends Bundle {
  val p = b

  val hit   = Bool()
  val pa    = UInt(p.bus.addressBits.W)
  val error = UInt(3.W)
}

@instantiable
class PreparedMap(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new Bundle {
    val reserve  = Flipped(Decoupled(new MapTag(b)))
    val prepared = Flipped(Decoupled(new PreparedSegment(b)))
    val ready    = Decoupled(new MapReady(b))
    val release  = Flipped(Decoupled(new MapTag(b)))
    val queries  = Input(Vec(2, new MapQuery(b)))
    val results  = Output(Vec(2, new MapResult(b)))
  })

  val slotBits   = log2Ceil(p.contexts)
  val live       = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val sealedList = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val notified   = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val tags       = Reg(Vec(p.contexts, UInt(p.idBits.W)))
  val counts     = RegInit(VecInit(Seq.fill(p.contexts)(0.U(log2Ceil(p.maxRanges + 1).W))))
  val errors     = RegInit(VecInit(Seq.fill(p.contexts)(Error.Ok.U(3.W))))
  val faultVA    = Reg(Vec(p.contexts, UInt(64.W)))
  val direction  = Reg(Vec(p.contexts, Bool()))
  val vas        = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(64.W))))
  val pas        = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(p.bus.addressBits.W))))
  val sizes      = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(13.W))))

  def matches(id: UInt): Vec[Bool] = VecInit((0 until p.contexts).map(i => live(i) && tags(i) === id))
  val free         = VecInit(live.map(!_))
  val reserveMatch = matches(io.reserve.bits.id)
  io.reserve.ready := free.asUInt.orR && !reserveMatch.asUInt.orR && !reset.asBool
  when(io.reserve.fire) {
    val slot = PriorityEncoder(free)
    live(slot)       := true.B; tags(slot)      := io.reserve.bits.id
    sealedList(slot) := false.B; notified(slot) := false.B
    counts(slot)     := 0.U; errors(slot)       := Error.Ok.U; faultVA(slot) := 0.U
  }

  val preparedMatch = matches(io.prepared.bits.id)
  val preparedSlot  = PriorityEncoder(preparedMatch)
  io.prepared.ready := preparedMatch.asUInt.orR && !sealedList(preparedSlot) && !reset.asBool
  val reservingPrepared = io.reserve.fire && io.reserve.bits.id === io.prepared.bits.id
  when(io.prepared.valid && !reset.asBool && !reservingPrepared) {
    assert(preparedMatch.asUInt.orR, "PreparedMap record has no reservation")
    assert(!sealedList(preparedSlot), "PreparedMap record follows the terminal record")
  }
  when(io.prepared.fire) {
    val record      = io.prepared.bits
    val lastVA      = record.va +& (record.bytes - 1.U)
    val lastPA      = record.pa +& (record.bytes - 1.U)
    val shapeOK     = record.bytes =/= 0.U && record.bytes <= 4096.U &&
      (record.va(11, 0) +& record.bytes) <= 4096.U &&
      (record.pa(11, 0) +& record.bytes) <= 4096.U &&
      !lastVA(64) && !lastPA(p.bus.addressBits)
    val capacityOK  = counts(preparedSlot) < p.maxRanges.U &&
      (counts(preparedSlot) =/= (p.maxRanges - 1).U || record.last)
    val directionOK = counts(preparedSlot) === 0.U || direction(preparedSlot) === record.write
    when(record.error =/= Error.Ok.U) {
      assert(record.last, "PreparedMap failure must be terminal")
      errors(preparedSlot)     := record.error; faultVA(preparedSlot) := record.va
      sealedList(preparedSlot) := true.B; counts(preparedSlot)        := 0.U
    }.otherwise {
      assert(shapeOK, "PreparedMap successful record must fit one VA and PA page")
      assert(capacityOK, "PreparedMap ranges must seal within capacity")
      assert(directionOK, "PreparedMap ranges must have one direction")
      when(!shapeOK || !capacityOK || !directionOK) {
        errors(preparedSlot)  := Mux(!capacityOK, Error.Capacity.U, Error.Shape.U)
        faultVA(preparedSlot) := record.va; sealedList(preparedSlot) := true.B
        counts(preparedSlot)  := 0.U
      }.otherwise {
        val index = counts(preparedSlot)(log2Ceil(p.maxRanges) - 1, 0)
        vas(preparedSlot)(index)                   := record.va; pas(preparedSlot)(index)   := record.pa
        sizes(preparedSlot)(index)                 := record.bytes; direction(preparedSlot) := record.write
        counts(preparedSlot)                       := counts(preparedSlot) + 1.U
        when(record.last)(sealedList(preparedSlot) := true.B)
      }
    }
  }

  val offer       = RegInit(false.B)
  val offeredSlot = Reg(UInt(slotBits.W))
  val awaiting    = VecInit((0 until p.contexts).map(i => live(i) && sealedList(i) && !notified(i)))
  when(!offer && awaiting.asUInt.orR) {
    offer := true.B; offeredSlot := PriorityEncoder(awaiting)
  }
  io.ready.valid := offer && !reset.asBool
  io.ready.bits.id    := tags(offeredSlot)
  io.ready.bits.error := errors(offeredSlot)
  io.ready.bits.va    := faultVA(offeredSlot)
  when(io.ready.fire) { notified(offeredSlot) := true.B; offer := false.B }

  val releaseMatch = matches(io.release.bits.id)
  val releaseSlot = PriorityEncoder(releaseMatch)
  io.release.ready := releaseMatch.asUInt.orR && notified(releaseSlot) && !reset.asBool
  when(io.release.fire) {
    live(releaseSlot)     := false.B; sealedList(releaseSlot) := false.B
    notified(releaseSlot) := false.B; counts(releaseSlot)     := 0.U
    errors(releaseSlot)   := Error.Ok.U
  }

  // A lookup takes three cycles and the query must be held throughout: the owning context is
  // registered, then the first covering range, and the third cycle answers. The owning context
  // must not change while it is queried.
  for (port <- 0 until 2) {
    val query = io.queries(port)
    val owner = matches(query.id)
    val owned = RegNext(owner.asUInt.orR)
    val slot  = RegNext(PriorityEncoder(owner))
    val last  = query.va +& (query.bytes - 1.U)
    def covers(context: UInt, index: Int): Bool =
      index.U < counts(context) && query.va >= vas(context)(index) &&
        last < (vas(context)(index) +& sizes(context)(index)) && direction(context) === query.write
    val hits       = VecInit((0 until p.maxRanges).map(covers(slot, _)))
    val covered    = RegNext(hits.asUInt.orR)
    val index      = RegNext(PriorityEncoder(hits))
    val retiring   = io.release.fire && io.release.bits.id === query.id
    val shapeOK    = query.bytes =/= 0.U && !last(64) &&
      (query.va(11, 0) +& query.bytes) <= 4096.U
    val authorized = owned && sealedList(slot) && notified(slot) &&
      errors(slot) === Error.Ok.U && !retiring && !reset.asBool
    val translated = pas(slot)(index) +& (query.va - vas(slot)(index))
    val hit        = query.valid && shapeOK && authorized && covered
    io.results(port).hit   := hit
    io.results(port).pa    := Mux(hit, translated(p.bus.addressBits - 1, 0), 0.U)
    io.results(port).error := Mux(!query.valid || hit, Error.Ok.U, Mux(!shapeOK, Error.Shape.U, Error.AccessFault.U))

    // Map invariants, checked on the query of this cycle against its current owner.
    val present        = PriorityEncoder(owner)
    val presentHits    = VecInit((0 until p.maxRanges).map(covers(present, _)))
    val presentAllowed = owner.asUInt.orR && sealedList(present) && notified(present) &&
      errors(present) === Error.Ok.U && !retiring && !reset.asBool
    val presentPA      = pas(present)(PriorityEncoder(presentHits)) +&
      (query.va - vas(present)(PriorityEncoder(presentHits)))
    when(query.valid && shapeOK && presentAllowed && presentHits.asUInt.orR) {
      assert(!presentPA(presentPA.getWidth - 1, p.bus.addressBits).orR, "PreparedMap translated address overflows PA")
      for (other <- 0 until p.maxRanges) {
        when(presentHits(other)) {
          assert(
            (pas(present)(other) +& (query.va - vas(present)(other))) === presentPA,
            "PreparedMap overlapping records disagree on PA"
          )
        }
      }
    }
  }
}
