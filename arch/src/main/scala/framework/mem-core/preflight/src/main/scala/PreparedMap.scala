package memcore.memory.preflight

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class MapTag(p: Params) extends Bundle { val id = UInt(p.idBits.W) }

class MapReady(p: Params) extends Bundle {
  val id    = UInt(p.idBits.W)
  val error = UInt(3.W)
  val va    = UInt(64.W)
}

class MapQuery(p: Params) extends Bundle {
  val valid = Bool()
  val id    = UInt(p.idBits.W)
  val va    = UInt(64.W)
  val bytes = UInt(13.W)
  val write = Bool()
}

class MapResult(p: Params) extends Bundle {
  val hit   = Bool()
  val pa    = UInt(p.bus.addressBits.W)
  val error = UInt(3.W)
}

@instantiable
class PreparedMap(p: Params) extends Module {

  @public val io = IO(new Bundle {
    val reserve  = Flipped(Decoupled(new MapTag(p)))
    val prepared = Flipped(Decoupled(new PreparedSegment(p)))
    val ready    = Decoupled(new MapReady(p))
    val release  = Flipped(Decoupled(new MapTag(p)))
    val queries  = Input(Vec(2, new MapQuery(p)))
    val results  = Output(Vec(2, new MapResult(p)))
  })

  private val slotBits = log2Ceil(p.contexts)
  val live             = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val sealedList       = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val notified         = RegInit(VecInit(Seq.fill(p.contexts)(false.B)))
  val tags             = Reg(Vec(p.contexts, UInt(p.idBits.W)))
  val counts           = RegInit(VecInit(Seq.fill(p.contexts)(0.U(log2Ceil(p.maxRanges + 1).W))))
  val errors           = RegInit(VecInit(Seq.fill(p.contexts)(Error.Ok.U(3.W))))
  val faultVA          = Reg(Vec(p.contexts, UInt(64.W)))
  val direction        = Reg(Vec(p.contexts, Bool()))
  val vas              = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(64.W))))
  val pas              = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(p.bus.addressBits.W))))
  val sizes            = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(13.W))))

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

  for (port <- 0 until 2) {
    val query      = io.queries(port)
    val owner      = matches(query.id)
    val slot       = PriorityEncoder(owner)
    val retiring   = io.release.fire && io.release.bits.id === query.id
    val last       = query.va +& (query.bytes - 1.U)
    val shapeOK    = query.bytes =/= 0.U && !last(64) &&
      (query.va(11, 0) +& query.bytes) <= 4096.U
    val authorized = owner.asUInt.orR && sealedList(slot) && notified(slot) &&
      errors(slot) === Error.Ok.U && !retiring && !reset.asBool
    val hits       = VecInit((0 until p.maxRanges).map { index =>
      index.U < counts(slot) && query.va >= vas(slot)(index) &&
      last < (vas(slot)(index) +& sizes(slot)(index)) && direction(slot) === query.write
    })
    val index      = PriorityEncoder(hits)
    val translated = pas(slot)(index) +& (query.va - vas(slot)(index))
    val hit        = query.valid && shapeOK && authorized && hits.asUInt.orR
    io.results(port).hit   := hit
    io.results(port).pa    := Mux(hit, translated(p.bus.addressBits - 1, 0), 0.U)
    io.results(port).error := Mux(!query.valid || hit, Error.Ok.U, Mux(!shapeOK, Error.Shape.U, Error.AccessFault.U))
    when(hit) {
      assert(!translated(translated.getWidth - 1, p.bus.addressBits).orR, "PreparedMap translated address overflows PA")
      for (other <- 0 until p.maxRanges) {
        when(hits(other)) {
          assert(
            (pas(slot)(other) +& (query.va - vas(slot)(other))) === translated,
            "PreparedMap overlapping records disagree on PA"
          )
        }
      }
    }
  }
}
