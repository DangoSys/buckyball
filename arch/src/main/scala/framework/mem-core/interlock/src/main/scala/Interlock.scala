package memcore.memory.interlock

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Register-only ownership table. Cache/DMA endpoints share reset and own the actual data. */
@instantiable
class Interlock(p: Params) extends Module {

  @public
  val io = IO(new Bundle {
    val dispatch      = Flipped(Decoupled(new Dispatch(p)))
    val cancel        = Flipped(Decoupled(new Dispatch(p)))
    val accessInfo    = Flipped(Decoupled(new AccessInfo(p)))
    val cpuQuery      = Input(new CpuQuery(p))
    val cpuAllow      = Output(Bool())
    // cpuAllow without the same-cycle dispatch term, for a CPU access that cannot share its cycle
    // with a dispatch (the access and the dispatching command occupy the same core stage).
    val cpuProbeAllow = Output(Bool())
    val maintenance   = Decoupled(new Maintenance(p))
    val maintained    = Flipped(Decoupled(new Acknowledgement(p)))
    val grant         = Decoupled(new Tag(p))
    val done          = Flipped(Decoupled(new Acknowledgement(p)))
    val complete      = Decoupled(new Tag(p))
  })

  private val indexBits                                                                                        = log2Ceil(p.entries)
  val unknown :: ready :: preWait :: granted :: running :: postReady :: postWait :: completed :: failed :: Nil = Enum(9)
  val live                                                                                                     = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val ids                                                                                                      = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.idBits.W))))
  val phase                                                                                                    = RegInit(VecInit(Seq.fill(p.entries)(unknown)))
  val memory                                                                                                   = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val writes                                                                                                   = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val first                                                                                                    = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.addressBits.W))))
  val last                                                                                                     = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.addressBits.W))))
  val rangeCount                                                                                               = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.rangeCountBits.W))))
  val rangeCursor                                                                                              = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.rangeIndexBits.W))))
  val rangeFirst                                                                                               = RegInit(VecInit(Seq.fill(p.entries)(VecInit(Seq.fill(p.maxRanges)(0.U(p.addressBits.W))))))
  val rangeLast                                                                                                = RegInit(VecInit(Seq.fill(p.entries)(VecInit(Seq.fill(p.maxRanges)(0.U(p.addressBits.W))))))
  val older                                                                                                    = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.entries.W))))
  val protocolError                                                                                            = RegInit(false.B)
  val lineMask                                                                                                 = (~(BigInt(p.lineBytes) - 1) & ((BigInt(1) << p.addressBits) - 1)).U(p.addressBits.W)
  def overlap(a: Int, b: Int): Bool = first(a) <= last(b) && first(b) <= last(a)

  def select(candidates: Vec[Bool], cursor: UInt): UInt = {
    MuxCase(
      0.U(indexBits.W),
      (0 until p.entries).map { offset =>
        val index = (cursor + offset.U)(indexBits - 1, 0)
        candidates(index) -> index
      }
    )
  }

  val free      = VecInit(live.map(!_))
  val freeIndex = PriorityEncoder(free.asUInt)
  val duplicate = VecInit((0 until p.entries).map(i => live(i) && ids(i) === io.dispatch.bits.id)).asUInt.orR
  io.dispatch.ready := free.asUInt.orR && !duplicate && !protocolError &&
    !(io.cancel.valid && io.cancel.bits.id === io.dispatch.bits.id)
  when(io.dispatch.valid && duplicate && !(io.cancel.valid && io.cancel.bits.id === io.dispatch.bits.id)) {
    assert(false.B, "Interlock dispatch ID is already live")
    protocolError := true.B
  }
  val cancelMatches = VecInit((0 until p.entries).map(i => live(i) && ids(i) === io.cancel.bits.id))
  val cancelFound = cancelMatches.asUInt.orR
  val cancelIndex = PriorityEncoder(cancelMatches.asUInt)

  val cancelConflict = (io.accessInfo.valid && io.accessInfo.bits.id === io.cancel.bits.id) ||
    (io.dispatch.valid && io.dispatch.bits.id === io.cancel.bits.id) ||
    (io.done.valid && io.done.bits.tag === io.cancel.bits.id)

  io.cancel.ready := cancelFound && phase(cancelIndex) === unknown && !cancelConflict && !protocolError
  when(io.cancel.valid) {
    when(cancelConflict) {
      assert(false.B, "Interlock cancel conflicts with the same command")
      protocolError := true.B
    }.otherwise {
      assert(cancelFound, "Interlock cancel ID has no reservation")
      when(cancelFound)(assert(phase(cancelIndex) === unknown, "Interlock cancel requires an unsealed reservation"))
      when(!cancelFound || phase(cancelIndex) =/= unknown)(protocolError := true.B)
    }
  }
  when(io.cancel.fire) {
    live(cancelIndex)        := false.B
    memory(cancelIndex)      := false.B
    rangeCount(cancelIndex)  := 0.U
    rangeCursor(cancelIndex) := 0.U
  }
  val infoMatches = VecInit((0 until p.entries).map(i => live(i) && ids(i) === io.accessInfo.bits.id))
  val infoIndex = PriorityEncoder(infoMatches.asUInt)
  val infoFound = infoMatches.asUInt.orR
  val newInfo   = io.dispatch.fire && io.dispatch.bits.id === io.accessInfo.bits.id
  io.accessInfo.ready := infoFound && phase(infoIndex) === unknown && !protocolError &&
    !(io.cancel.valid && io.cancel.bits.id === io.accessInfo.bits.id)
  when(io.accessInfo.valid && !(io.cancel.valid && io.cancel.bits.id === io.accessInfo.bits.id)) {
    assert(infoFound || newInfo, "Interlock access info ID has no reservation")
    when(infoFound)(assert(phase(infoIndex) === unknown, "Interlock access info is already declared"))
    when(!infoFound && !newInfo || infoFound && phase(infoIndex) =/= unknown)(protocolError := true.B)
  }
  val end = io.accessInfo.bits.base.pad(p.addressBits + 1) +& (io.accessInfo.bits.bytes - 1.U)
  when(io.accessInfo.fire) {
    when(io.accessInfo.bits.hasMemory) {
      val count      = rangeCount(infoIndex)
      val consistent = count === 0.U || writes(infoIndex) === io.accessInfo.bits.write
      val capacity   = count < p.maxRanges.U && (count =/= (p.maxRanges - 1).U || io.accessInfo.bits.last)
      assert(io.accessInfo.bits.bytes =/= 0.U, "Interlock memory range is empty")
      when(io.accessInfo.bits.bytes =/= 0.U) {
        assert(!end(p.addressBits + 1, p.addressBits).orR, "Interlock memory range exceeds physical width")
      }
      assert(consistent, "Interlock ranges must have one access direction")
      assert(capacity, "Interlock range list must seal within capacity")
      when(io.accessInfo.bits.bytes === 0.U || end(p.addressBits + 1, p.addressBits).orR || !consistent || !capacity) {
        phase(infoIndex) := failed
        protocolError    := true.B
      }.otherwise {
        val index  = if (p.maxRanges == 1) 0.U else count(p.rangeIndexBits - 1, 0)
        val begin  = io.accessInfo.bits.base & lineMask
        val finish = end(p.addressBits - 1, 0) & lineMask
        rangeFirst(infoIndex)(index) := begin
        rangeLast(infoIndex)(index)  := finish
        rangeCount(infoIndex)        := count + 1.U
        memory(infoIndex)            := true.B
        writes(infoIndex)            := io.accessInfo.bits.write
        first(infoIndex)             := Mux(count === 0.U || begin < first(infoIndex), begin, first(infoIndex))
        last(infoIndex)              := Mux(count === 0.U || finish > last(infoIndex), finish, last(infoIndex))
        when(io.accessInfo.bits.last) { phase(infoIndex) := ready; rangeCursor(infoIndex) := 0.U }
      }
    }.otherwise {
      val legal = rangeCount(infoIndex) === 0.U && io.accessInfo.bits.last
      assert(legal, "Interlock no-memory declaration must seal an empty command")
      when(legal) {
        memory(infoIndex) := false.B
        phase(infoIndex)  := completed
      }.otherwise {
        phase(infoIndex) := failed
        protocolError    := true.B
      }
    }
  }

  val queryLine = io.cpuQuery.paddr & lineMask
  val alignMask = MuxLookup(io.cpuQuery.sizeLog2, 0.U(3.W))(Seq(0.U -> 0.U, 1.U -> 1.U, 2.U -> 3.U, 3.U -> 7.U))
  when(io.cpuQuery.valid) {
    assert(io.cpuQuery.sizeLog2 <= 3.U, "Interlock CPU size must be 1, 2, 4 or 8 bytes")
    when(io.cpuQuery.sizeLog2 <= 3.U) {
      assert((io.cpuQuery.paddr(2, 0) & alignMask) === 0.U, "Interlock CPU access must be naturally aligned")
    }
  }

  val blocked = (0 until p.entries).map { i =>
    val hit = (0 until p.maxRanges).map(j =>
      j.U < rangeCount(i) &&
        queryLine >= rangeFirst(i)(j) && queryLine <= rangeLast(i)(j)
    ).reduce(_ || _)
    live(i) && (phase(i) === unknown || memory(i) && hit &&
      (writes(i) || io.cpuQuery.write || phase(i) === failed))
  }.reduce(_ || _)

  io.cpuAllow      := !io.cpuQuery.valid || (!blocked && !io.cpuQuery.olderDispatchPending && !io.dispatch.fire && !protocolError)
  io.cpuProbeAllow := !io.cpuQuery.valid || (!blocked && !io.cpuQuery.olderDispatchPending && !protocolError)

  val maintenanceCursor  = RegInit(0.U(indexBits.W))
  val maintenanceOffer   = RegInit(false.B)
  val maintenanceIndex   = RegInit(0.U(indexBits.W))
  val maintenancePayload = RegInit(0.U.asTypeOf(new Maintenance(p)))
  val maintenanceBusy    = RegInit(false.B)
  val maintenanceOwner   = RegInit(0.U(indexBits.W))

  val maintenanceCandidates = VecInit((0 until p.entries).map(i =>
    live(i) &&
      (phase(i) === ready && older(i) === 0.U || phase(i) === postReady)
  ))

  when(!maintenanceOffer && !maintenanceBusy && maintenanceCandidates.asUInt.orR && !protocolError) {
    val index = select(maintenanceCandidates, maintenanceCursor)
    maintenanceOffer             := true.B
    maintenanceIndex             := index
    maintenancePayload.tag       := ids(index)
    maintenancePayload.firstLine := rangeFirst(index)(rangeCursor(index))
    maintenancePayload.lastLine  := rangeLast(index)(rangeCursor(index))
    maintenancePayload.op        := Mux(
      phase(index) === postReady,
      MaintenanceOp.Invalidate.U,
      Mux(writes(index), MaintenanceOp.CleanInvalidate.U, MaintenanceOp.Clean.U)
    )
  }
  io.maintenance.valid := maintenanceOffer && !protocolError
  io.maintenance.bits  := maintenancePayload
  when(io.maintenance.fire) {
    maintenanceOffer        := false.B
    maintenanceBusy         := true.B
    maintenanceOwner        := maintenanceIndex
    maintenanceCursor       := maintenanceIndex + 1.U
    phase(maintenanceIndex) := Mux(maintenancePayload.op === MaintenanceOp.Invalidate.U, postWait, preWait)
  }
  io.maintained.ready  := maintenanceBusy && !protocolError
  when(io.maintained.valid && !maintenanceBusy && !io.maintenance.fire) {
    assert(false.B, "Interlock maintenance response has no request")
    protocolError := true.B
  }
  when(io.maintained.fire) {
    assert(io.maintained.bits.tag === ids(maintenanceOwner), "Interlock maintenance response tag mismatch")
    assert(io.maintained.bits.ok, "Interlock maintenance failed")
    maintenanceBusy := false.B
    when(io.maintained.bits.tag =/= ids(maintenanceOwner)) {
      protocolError           := true.B
      phase(maintenanceOwner) := failed
    }.elsewhen(!io.maintained.bits.ok) { phase(maintenanceOwner) := failed; protocolError := true.B }
      .otherwise {
        val next = rangeCursor(maintenanceOwner) +& 1.U
        val post = phase(maintenanceOwner) === postWait
        when(next < rangeCount(maintenanceOwner)) {
          rangeCursor(maintenanceOwner) := next
          phase(maintenanceOwner)       := Mux(post, postReady, ready)
        }.otherwise(phase(maintenanceOwner) := Mux(post, completed, granted))
      }
  }

  val grantCursor     = RegInit(0.U(indexBits.W))
  val grantOffer      = RegInit(false.B)
  val grantIndex      = RegInit(0.U(indexBits.W))
  val grantPayload    = RegInit(0.U.asTypeOf(new Tag(p)))
  val grantCandidates = VecInit((0 until p.entries).map(i => live(i) && phase(i) === granted))
  when(!grantOffer && grantCandidates.asUInt.orR && !protocolError) {
    val index = select(grantCandidates, grantCursor)
    grantOffer := true.B; grantIndex := index; grantPayload.tag := ids(index)
  }
  io.grant.valid := grantOffer && !protocolError
  io.grant.bits := grantPayload
  when(io.grant.fire) {
    grantOffer := false.B; grantCursor := grantIndex + 1.U; phase(grantIndex) := running
  }
  val doneMatches = VecInit((0 until p.entries).map(i => live(i) && ids(i) === io.done.bits.tag))
  val doneIndex = PriorityEncoder(doneMatches.asUInt)
  val doneFound = doneMatches.asUInt.orR
  val newDone   = io.grant.fire && io.grant.bits.tag === io.done.bits.tag
  io.done.ready := doneFound && phase(doneIndex) === running && !protocolError &&
    !(io.cancel.valid && io.cancel.bits.id === io.done.bits.tag)
  when(io.done.valid && !(io.cancel.valid && io.cancel.bits.id === io.done.bits.tag)) {
    assert(doneFound, "Interlock DMA done tag has no reservation")
    when(doneFound) {
      assert(phase(doneIndex) === running || newDone, "Interlock DMA done requires a granted access")
    }
    when(!doneFound || doneFound && phase(doneIndex) =/= running && !newDone)(protocolError := true.B)
  }
  when(io.done.fire) {
    rangeCursor(doneIndex)               := 0.U
    assert(io.done.bits.ok, "Interlock DMA failed")
    when(!io.done.bits.ok)(protocolError := true.B)
    phase(doneIndex)                     := Mux(!io.done.bits.ok, failed, Mux(writes(doneIndex), postReady, completed))
  }

  val completeCursor     = RegInit(0.U(indexBits.W))
  val completeOffer      = RegInit(false.B)
  val completeIndex      = RegInit(0.U(indexBits.W))
  val completePayload    = RegInit(0.U.asTypeOf(new Tag(p)))
  val completeCandidates = VecInit((0 until p.entries).map(i => live(i) && phase(i) === completed))
  when(!completeOffer && completeCandidates.asUInt.orR && !protocolError) {
    val index = select(completeCandidates, completeCursor)
    completeOffer := true.B; completeIndex := index; completePayload.tag := ids(index)
  }
  io.complete.valid := completeOffer && !protocolError
  io.complete.bits := completePayload
  when(io.complete.fire) {
    completeOffer := false.B; completeCursor := completeIndex + 1.U; live(completeIndex) := false.B
  }
  for (i <- 0 until p.entries) {
    val retained = VecInit((0 until p.entries).map { j =>
      older(i)(j) && live(j) && !(io.complete.fire && completeIndex === j.U) &&
      !(io.cancel.fire && cancelIndex === j.U) &&
      (phase(i) === unknown || phase(j) === unknown ||
        memory(i) && memory(j) && overlap(i, j) && (writes(i) || writes(j) || phase(j) === failed))
    }).asUInt
    older(i) := retained
  }
  when(io.dispatch.fire) {
    live(freeIndex)        := true.B
    ids(freeIndex)         := io.dispatch.bits.id
    phase(freeIndex)       := unknown
    rangeCount(freeIndex)  := 0.U
    rangeCursor(freeIndex) := 0.U
    memory(freeIndex)      := false.B
    writes(freeIndex)      := false.B
    first(freeIndex)       := 0.U; last(freeIndex) := 0.U
    older(freeIndex)       := VecInit((0 until p.entries).map(i =>
      live(i) &&
        !(io.complete.fire && completeIndex === i.U) && !(io.cancel.fire && cancelIndex === i.U)
    )).asUInt
  }
}
