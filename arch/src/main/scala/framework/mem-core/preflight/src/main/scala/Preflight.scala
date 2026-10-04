package memcore.memory.preflight

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.bus.chi.rnf.{CacheAccess, CacheResult}
import memcore.memory.mmu.Walker

/** Preparation only. A consumer retains the entire successful mapping until DMA retires. */
@instantiable
class Preflight(p: Params) extends Module {

  @public val io = IO(new Bundle {
    val command       = Flipped(Decoupled(new Command(p)))
    val prepared      = Decoupled(new PreparedSegment(p))
    val authorization = Decoupled(new Authorization(p))
    val permission    = Flipped(Decoupled(new Permission(p)))
    val pteRequest    = Decoupled(new CacheAccess(p.bus))
    val pteResponse   = Flipped(Decoupled(new CacheResult))
  })

  val free :: queued :: working :: ready :: Nil = Enum(4)
  val phase                                     = RegInit(VecInit(Seq.fill(p.contexts)(free)))
  val commands                                  = Reg(Vec(p.contexts, new Command(p)))
  val counts                                    = RegInit(VecInit(Seq.fill(p.contexts)(0.U(4.W))))
  val errors                                    = Reg(Vec(p.contexts, UInt(3.W)))
  val faultVA                                   = Reg(Vec(p.contexts, UInt(64.W)))
  val vas                                       = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(64.W))))
  val pas                                       = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(p.bus.addressBits.W))))
  val sizes                                     = Reg(Vec(p.contexts, Vec(p.maxRanges, UInt(13.W))))
  val available                                 = VecInit(phase.map(_ === free))
  val duplicate                                 =
    (0 until p.contexts).map(i => phase(i) =/= free && commands(i).id === io.command.bits.id).reduce(_ || _)
  io.command.ready := available.asUInt.orR && !duplicate && !reset.asBool
  when(io.command.fire) {
    val slot = PriorityEncoder(available.asUInt)
    commands(slot) := io.command.bits; counts(slot) := 0.U; errors(slot) := Error.Ok.U; phase(slot) := queued
  }

  def select(candidates: Vec[Bool], cursor: UInt): UInt = {
    MuxCase(
      0.U(2.W),
      (0 until p.contexts).map { off =>
        val index = (cursor + off.U)(1, 0)
        candidates(index) -> index
      }
    )
  }

  val idle :: checkShape :: span :: sendSpan :: segment :: walk :: waitWalk :: authorize :: waitPermission :: save :: Nil =
    Enum(10)
  val state                                                                                                               = RegInit(idle)
  val worker                                                                                                              = Reg(UInt(2.W))
  val cursor                                                                                                              = RegInit(0.U(2.W))
  val row                                                                                                                 = Reg(UInt(32.W)); val column      = Reg(UInt(32.W))
  val dense                                                                                                               = Reg(Bool()); val wholeBytes      = Reg(UInt(32.W))
  val spanAddress                                                                                                         = Reg(UInt(64.W)); val spanBytes   = Reg(UInt(32.W))
  val spanLast                                                                                                            = Reg(Bool())
  val currentVA                                                                                                           = Reg(UInt(64.W)); val currentPA   = Reg(UInt(p.bus.addressBits.W))
  val currentBytes                                                                                                        = Reg(UInt(13.W)); val segmentLast = Reg(Bool())
  val cancelSegmenter                                                                                                     = WireDefault(false.B)
  val splitter                                                                                                            = Instantiate(new Segmenter(4096, 4096))
  splitter.io.cancel := cancelSegmenter
  val walker = Instantiate(new Walker(p.bus))
  val cmd    = commands(worker)

  def finish(code: UInt, address: UInt): Unit = {
    errors(worker) := code; faultVA(worker) := address; phase(worker) := ready; state := idle
    when(code =/= Error.Ok.U) { counts(worker) := 0.U; cancelSegmenter := true.B }
  }

  val waiting = VecInit(phase.map(_ === queued))
  when(state === idle && waiting.asUInt.orR) {
    val slot = select(waiting, cursor)
    worker := slot; cursor := slot + 1.U; phase(slot) := working; state := checkShape
  }
  when(state === checkShape) {
    val rowExtent = cmd.columns * cmd.spanBytes
    val total     = cmd.rows * rowExtent
    val isDense   = (cmd.columns === 1.U || cmd.columnStride === cmd.spanBytes) &&
      (cmd.rows === 1.U || cmd.rowStride === rowExtent)
    val end       = cmd.baseVA.pad(67) + ((cmd.rows - 1.U) * cmd.rowStride).pad(67) +
      ((cmd.columns - 1.U) * cmd.columnStride).pad(67) + cmd.spanBytes.pad(67) - 1.U
    when(cmd.rows === 0.U || cmd.columns === 0.U || cmd.spanBytes === 0.U ||
      (cmd.rows > 1.U && cmd.rowStride === 0.U) || (cmd.columns > 1.U && cmd.columnStride === 0.U)) {
      finish(Error.Shape.U, cmd.baseVA)
    }.elsewhen(cmd.mode =/= 0.U && cmd.mode =/= 8.U || cmd.privilege === 2.U) {
      finish(Error.Context.U, cmd.baseVA)
    }.elsewhen(end(66, 64).orR || (isDense && total > "hffffffff".U)) {
      finish(Error.Overflow.U, cmd.baseVA)
    }.otherwise {
      dense := isDense; wholeBytes := total(31, 0); row := 0.U; column := 0.U; state := span
    }
  }
  when(state === span) {
    val start  = cmd.baseVA.pad(67) + (row * cmd.rowStride).pad(67) + (column * cmd.columnStride).pad(67)
    val bytes  = Mux(dense, wholeBytes, cmd.spanBytes)
    val mask   = (~(BigInt(p.beatBytes) - 1) & ((BigInt(1) << 67) - 1)).U(67.W)
    val first  = Mux(cmd.write, start, start & mask)
    val end    = Mux(cmd.write, start + bytes.pad(67), (start + bytes.pad(67) + (p.beatBytes - 1).U) & mask)
    val length = end - first
    when(length > "hffffffff".U)(finish(Error.Overflow.U, start(63, 0)))
      .otherwise {
        spanAddress := first(63, 0); spanBytes := length(31, 0)
        spanLast    := dense || (row === cmd.rows - 1.U && column === cmd.columns - 1.U)
        state       := sendSpan
      }
  }
  splitter.io.descriptor.valid := state === sendSpan
  splitter.io.descriptor.bits.addr        := spanAddress
  splitter.io.descriptor.bits.bytes       := spanBytes
  splitter.io.descriptor.bits.rowBytes    := spanBytes
  splitter.io.descriptor.bits.rowStride   := spanBytes
  when(splitter.io.descriptor.fire)(state := segment)
  splitter.io.segment.ready               := state === segment
  when(splitter.io.segment.fire) {
    currentVA   := splitter.io.segment.bits.addr; currentBytes := splitter.io.segment.bits.bytes(12, 0)
    segmentLast := splitter.io.segment.bits.last; state        := walk
  }
  walker.io.config.mode                   := cmd.mode; walker.io.config.rootPpn    := cmd.rootPpn
  walker.io.req.valid                     := state === walk
  walker.io.req.bits.vaddr                := currentVA; walker.io.req.bits.write   := cmd.write; walker.io.req.bits.execute := false.B
  walker.io.req.bits.privilege            := cmd.privilege; walker.io.req.bits.sum := cmd.sum; walker.io.req.bits.mxr       := cmd.mxr
  when(walker.io.req.fire)(state          := waitWalk)
  walker.io.resp.ready                    := state === waitWalk
  when(walker.io.resp.fire) {
    val end = walker.io.resp.bits.paddr.pad(p.bus.addressBits + 2) + currentBytes.pad(p.bus.addressBits + 2) - 1.U
    when(walker.io.resp.bits.pageFault)(finish(Error.PageFault.U, currentVA))
      .elsewhen(walker.io.resp.bits.accessFault || end(p.bus.addressBits + 1, p.bus.addressBits).orR) {
        finish(Error.AccessFault.U, currentVA)
      }.otherwise { currentPA := walker.io.resp.bits.paddr; state := authorize }
  }
  val pteIdle :: pteAsk :: pteWait :: pteIssue :: pteReturn :: pteReject :: pteRejectReturn :: Nil = Enum(7)
  val pteState = RegInit(pteIdle)
  when(pteState === pteIdle && walker.io.access.valid)(pteState := pteAsk)
  io.authorization.valid                                        := (pteState === pteAsk || state === authorize) && !reset.asBool
  io.authorization.bits.id                                      := cmd.id
  io.authorization.bits.isPte                                   := pteState === pteAsk
  io.authorization.bits.pa                                      := Mux(pteState === pteAsk, walker.io.access.bits.addr, currentPA)
  io.authorization.bits.bytes                                   := Mux(pteState === pteAsk, 8.U, currentBytes)
  io.authorization.bits.write                                   := pteState =/= pteAsk && cmd.write
  io.authorization.bits.privilege                               := Mux(pteState === pteAsk, 1.U, cmd.privilege)
  when(io.authorization.fire) {
    when(pteState === pteAsk)(pteState := pteWait).otherwise(state := waitPermission)
  }
  io.permission.ready                                           := (pteState === pteWait || state === waitPermission) && !reset.asBool
  when(io.permission.fire) {
    assert(io.permission.bits.id === cmd.id, "Preflight permission response tag mismatch")
    when(pteState === pteWait)(pteState := Mux(io.permission.bits.allow, pteIssue, pteReject))
      .otherwise {
        when(io.permission.bits.allow)(state := save).otherwise(finish(Error.AccessFault.U, currentVA))
      }
  }
  io.pteRequest.valid                                           := pteState === pteIssue && walker.io.access.valid && !reset.asBool
  io.pteRequest.bits                                            := walker.io.access.bits
  walker.io.access.ready                                        := pteState === pteReject || pteState === pteIssue && io.pteRequest.ready
  when(walker.io.access.fire)(pteState                          := Mux(pteState === pteReject, pteRejectReturn, pteReturn))
  walker.io.result.valid                                        := pteState === pteRejectReturn || pteState === pteReturn && io.pteResponse.valid
  walker.io.result.bits                                         := 0.U.asTypeOf(walker.io.result.bits)
  walker.io.result.bits.data                                    := Mux(pteState === pteRejectReturn, 0.U, io.pteResponse.bits.data)
  walker.io.result.bits.error                                   := pteState === pteRejectReturn || io.pteResponse.bits.error
  io.pteResponse.ready                                          := pteState === pteReturn && walker.io.result.ready && !reset.asBool
  when(walker.io.result.fire)(pteState                          := pteIdle)
  // A segment that starts inside or right after the previous range of the same page extends it:
  // one page shares one translation, so its PA stays contiguous. Strided shapes such as 2-D tiles
  // then need one range per row and page instead of one per element.
  val previous     = (counts(worker) - 1.U)(2, 0)
  val previousVA   = vas(worker)(previous)
  val previousEnd  = previousVA +& sizes(worker)(previous)
  val currentEnd   = currentVA +& currentBytes
  val merge        = counts(worker) =/= 0.U && previousVA(63, 12) === currentVA(63, 12) &&
    currentVA >= previousVA && currentVA <= previousEnd
  val saveCapacity = merge || counts(worker) =/= p.maxRanges.U
  when(state === save && !saveCapacity)(finish(Error.Capacity.U, currentVA))
  when(state === save && saveCapacity) {
    when(merge) {
      assert(
        pas(worker)(previous) +& (currentVA - previousVA) === currentPA,
        "Preflight same-page ranges disagree on PA"
      )
      when(currentEnd > previousEnd)(sizes(worker)(previous) := currentEnd - previousVA)
    }.otherwise {
      val index = counts(worker)(2, 0)
      vas(worker)(index) := currentVA; pas(worker)(index) := currentPA; sizes(worker)(index) := currentBytes
      counts(worker)     := counts(worker) + 1.U
    }
    when(segmentLast && spanLast)(finish(Error.Ok.U, currentVA))
      .elsewhen(segmentLast) {
        when(column === cmd.columns - 1.U) { column := 0.U; row := row + 1.U }
          .otherwise(column := column + 1.U)
        state               := span
      }.otherwise(state := segment)
  }
  val offer        = RegInit(false.B); val outputSlot = Reg(UInt(2.W)); val outputIndex = Reg(UInt(3.W))
  val outputCursor = RegInit(0.U(2.W))
  val prepared     = VecInit(phase.map(_ === ready))
  when(!offer && prepared.asUInt.orR) {
    offer := true.B; outputSlot := select(prepared, outputCursor); outputIndex := 0.U
  }
  io.prepared.valid := offer && !reset.asBool
  val failed       = errors(outputSlot) =/= Error.Ok.U
  io.prepared.bits.id    := commands(outputSlot).id
  io.prepared.bits.va    := Mux(failed, faultVA(outputSlot), vas(outputSlot)(outputIndex))
  io.prepared.bits.pa    := Mux(failed, 0.U, pas(outputSlot)(outputIndex))
  io.prepared.bits.bytes := Mux(failed, 0.U, sizes(outputSlot)(outputIndex))
  io.prepared.bits.write := commands(outputSlot).write
  io.prepared.bits.error := errors(outputSlot)
  io.prepared.bits.last  := failed || outputIndex === counts(outputSlot) - 1.U
  when(io.prepared.fire) {
    when(io.prepared.bits.last) { phase(outputSlot) := free; offer := false.B; outputCursor := outputSlot + 1.U }
      .otherwise(outputIndex := outputIndex + 1.U)
  }
}
