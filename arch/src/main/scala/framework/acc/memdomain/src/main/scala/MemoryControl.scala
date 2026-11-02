package framework.acc.memdomain

import framework.acc.gpdomain.configs.GpDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.memory.MemCtrl
import framework.core.memory.access.preflight.{Command, Error, PmpEntry, PreparationPort}
import framework.acc.memdomain.toddr.{DmaError, DmaStatus}
import framework.acc.memdomain.tomem.ControlPort

/** Address-space state frozen with the instruction and consumed only by memory requests. */
class MemoryContext(permission: PmpParams) extends Bundle {
  val satp      = UInt(64.W)
  val privilege = UInt(2.W)
  val sum       = Bool()
  val mxr       = Bool()
  val pmp       =
    Vec(permission.entries, new PmpEntry(permission))
}

/** Tracks the three DMA producers, independent of frontend tags and ROB retirement. */
@instantiable
class MemoryControl(
  gpParam:         GpDomainParam,
  transportParam:  TransportParam,
  permissionParam: PmpParams,
  trackingParam:   TrackingParam,
  walkerParam:     WalkerParam)
    extends Module {
  val transport = transportParam
  val producers = if (gpParam.enable) 3 else 2
  require(transport.tracking.entries >= producers && transport.prepared.idBits >= log2Ceil(producers))

  @public
  val io = IO(new Bundle {
    val service     = new MemCtrl(trackingParam, walkerParam)
    val preparation = new PreparationPort(transportParam.mapping, permissionParam, walkerParam)
    val memory      = Flipped(new ControlPort(transportParam, permissionParam))
    val decisions   = Output(Vec(3, new framework.acc.memdomain.toddr.DmaDecision(transportParam)))
    val queries     = Input(Vec(2, new framework.core.memory.access.preflight.MapQuery(transportParam.mapping)))
    val mappings    = Output(Vec(2, new framework.core.memory.access.preflight.MapResult(transportParam.mapping)))
  })

  val coherence = chisel3.experimental.hierarchy.Instantiate(
    new framework.acc.memdomain.tol1.ToL1(transportParam.interlock)
  )

  val active            = RegInit(VecInit(Seq.fill(3)(false.B)))
  val dispatchPending   = RegInit(VecInit(Seq.fill(3)(false.B)))
  val preparePending    = RegInit(VecInit(Seq.fill(3)(false.B)))
  val mapReady          = RegInit(VecInit(Seq.fill(3)(false.B)))
  val granted           = RegInit(VecInit(Seq.fill(3)(false.B)))
  val doneIssued        = RegInit(VecInit(Seq.fill(3)(false.B)))
  val coherent          = RegInit(VecInit(Seq.fill(3)(false.B)))
  val released          = RegInit(VecInit(Seq.fill(3)(false.B)))
  val completionPending = RegInit(VecInit(Seq.fill(3)(false.B)))
  val completionStatus  = Reg(Vec(3, new DmaStatus))
  val contexts          = Reg(Vec(3, new MemoryContext(permissionParam)))
  val shapes            = Reg(Vec(3, new Command(transportParam.mapping)))
  val halted            = RegInit(false.B)
  val firstFault        = RegInit(0.U.asTypeOf(new DmaStatus))

  def stop(error: UInt, address: UInt): Unit = {
    when(!halted) {
      halted             := true.B
      firstFault.error   := error
      firstFault.address := address
    }
  }

  def dmaError(error: UInt): UInt = MuxLookup(error, DmaError.Context.U)(Seq(
    Error.Ok.U          -> DmaError.None.U,
    Error.Shape.U       -> DmaError.Shape.U,
    Error.Overflow.U    -> DmaError.Shape.U,
    Error.PageFault.U   -> DmaError.PageFault.U,
    Error.AccessFault.U -> DmaError.AccessFault.U,
    Error.Capacity.U    -> DmaError.Capacity.U,
    Error.Context.U     -> DmaError.Context.U
  ))

  // Producer 0 is the bank reader, 1 the writer, and 2 the image reader.
  for (i <- 0 until 3) {
    val access  = if (i == 1) io.memory.write.bits.access else io.memory.read.bits.access
    val offered =
      if (i >= producers) false.B
      else if (i == 1) io.memory.write.valid
      else io.memory.read.valid && io.memory.read.bits.producer === i.U
    when(offered && !active(i) && !halted) {
      active(i)              := true.B
      dispatchPending(i)     := true.B
      preparePending(i)      := false.B
      mapReady(i)            := false.B
      granted(i)             := false.B
      doneIssued(i)          := false.B
      coherent(i)            := false.B
      released(i)            := false.B
      contexts(i)            := access.context
      shapes(i).id           := i.U
      shapes(i).baseVA       := access.baseVA
      shapes(i).rows         := access.rows(31, 0)
      shapes(i).columns      := access.columns(31, 0)
      shapes(i).spanBytes    := access.spanBytes(31, 0)
      shapes(i).columnStride := access.columnStride(31, 0)
      shapes(i).rowStride    := access.rowStride(31, 0)
      shapes(i).write        := (i == 1).B
      shapes(i).mode         := access.context.satp(63, 60)
      shapes(i).rootPpn      := access.context.satp(43, 0)
      shapes(i).privilege    := access.context.privilege
      shapes(i).sum          := access.context.sum
      shapes(i).mxr          := access.context.mxr
      when(access.rows(63, 32).orR || access.columns(63, 32).orR || access.spanBytes(63, 32).orR ||
        access.columnStride(63, 32).orR || access.rowStride(63, 32).orR) {
        stop(DmaError.Shape.U, access.baseVA)
      }
    }
    io.decisions(i).parentId := i.U
    io.decisions(i).valid := active(i) && ((mapReady(i) && granted(i)) || halted)
    io.decisions(i).fault := firstFault
    val finished     = io.memory.finished(i)
    when(finished.valid && finished.bits.error =/= DmaError.None.U) {
      stop(finished.bits.error, finished.bits.address)
    }
    // A rejected image may finish locally without ever starting DMA.
    val localFailure = !active(i) && finished.bits.error =/= DmaError.None.U
    when(!completionPending(i) && finished.valid && (localFailure || (active(i) && (released(i) || halted)))) {
      completionPending(i) := true.B
      completionStatus(i)  := Mux(halted, firstFault, finished.bits)
    }
    io.memory.complete(i).valid := completionPending(i)
    io.memory.complete(i).bits := completionStatus(i)
    when(io.memory.complete(i).fire) {
      completionPending(i) := false.B
      active(i)            := false.B
    }
  }

  val dispatchOffer = RegInit(false.B)
  val dispatchSlot  = Reg(UInt(2.W))
  when(!dispatchOffer && dispatchPending.asUInt.orR && !halted) {
    dispatchOffer := true.B
    dispatchSlot  := PriorityEncoder(dispatchPending)
  }
  coherence.io.dispatch.valid := dispatchOffer && !halted
  coherence.io.dispatch.bits.id := dispatchSlot
  when(coherence.io.dispatch.fire) {
    dispatchOffer                 := false.B
    dispatchPending(dispatchSlot) := false.B
    preparePending(dispatchSlot)  := true.B
  }

  val prepareOffer = RegInit(false.B)
  val prepareSlot  = Reg(UInt(2.W))
  when(!prepareOffer && preparePending.asUInt.orR && !halted) {
    prepareOffer := true.B
    prepareSlot  := PriorityEncoder(preparePending)
  }
  io.preparation.command.valid := prepareOffer && !halted
  io.preparation.command.bits := shapes(prepareSlot)
  when(io.preparation.command.fire) {
    prepareOffer                := false.B
    preparePending(prepareSlot) := false.B
  }
  val contextId = io.preparation.contextTag
  io.preparation.contextValid := contextId < 3.U && active(contextId)
  io.preparation.contextId    := contextId
  io.preparation.contextPmp   := contexts(contextId).pmp

  val range = io.preparation.ranges
  val info  = coherence.io.accessInfo
  info.valid                 := range.valid && range.bits.error === Error.Ok.U && !halted
  info.bits.id               := range.bits.id
  info.bits.hasMemory        := true.B
  info.bits.base             := range.bits.pa
  info.bits.bytes            := range.bits.bytes
  info.bits.write            := range.bits.write
  info.bits.last             := range.bits.last
  range.ready                := range.bits.error =/= Error.Ok.U || (info.ready && !halted)
  when(range.fire && range.bits.error =/= Error.Ok.U) {
    stop(dmaError(range.bits.error), range.bits.va)
  }
  io.preparation.ready.ready := true.B
  when(io.preparation.ready.fire) {
    val result = io.preparation.ready.bits
    assert(result.id < 3.U && active(result.id), "Prepared result has no DMA owner")
    mapReady(result.id) := true.B
    when(result.error =/= Error.Ok.U)(stop(dmaError(result.error), result.va))
  }
  coherence.io.grant.ready   := true.B
  when(coherence.io.grant.fire) {
    val id = coherence.io.grant.bits.tag
    assert(id < 3.U && active(id), "Coherence grant has no DMA owner")
    granted(id) := true.B
  }
  io.preparation.queries     := io.queries
  io.mappings                := io.preparation.results

  // Finishing the bank stream precedes post-DMA cache maintenance and map release.
  val doneCandidates =
    VecInit((0 until 3).map(i => active(i) && io.memory.finished(i).valid && mapReady(i) && granted(i) && !doneIssued(i)))
  val doneOffer      = RegInit(false.B)
  val doneSlot       = Reg(UInt(2.W))
  when(!doneOffer && doneCandidates.asUInt.orR && !halted) {
    doneOffer := true.B
    doneSlot  := PriorityEncoder(doneCandidates)
  }
  coherence.io.done.valid := doneOffer && !halted
  coherence.io.done.bits.tag  := doneSlot
  coherence.io.done.bits.ok   := true.B
  when(coherence.io.done.fire) {
    doneOffer            := false.B
    doneIssued(doneSlot) := true.B
  }
  coherence.io.complete.ready := true.B
  when(coherence.io.complete.fire) {
    val id = coherence.io.complete.bits.tag
    assert(id < 3.U && active(id), "Coherence completion has no DMA owner")
    coherent(id) := true.B
  }
  val releaseCandidates = VecInit((0 until 3).map(i => active(i) && coherent(i) && !released(i)))
  val releaseOffer = RegInit(false.B)
  val releaseSlot  = Reg(UInt(2.W))
  when(!releaseOffer && releaseCandidates.asUInt.orR && !halted) {
    releaseOffer := true.B
    releaseSlot  := PriorityEncoder(releaseCandidates)
  }
  io.preparation.release.valid := releaseOffer && !halted
  io.preparation.release.bits.id := releaseSlot
  when(io.preparation.release.fire) {
    releaseOffer          := false.B
    released(releaseSlot) := true.B
  }

  coherence.io.cpuQuery     := io.service.cpuQuery
  io.service.cpuAllow       := coherence.io.cpuAllow && !halted
  io.service.maintenance <> coherence.io.maintenance
  coherence.io.maintained <> io.service.maintained
  io.service.pteRequest <> io.preparation.pteRequest
  io.preparation.pteResponse <> io.service.pteResponse
  coherence.io.cancel.valid := false.B
  coherence.io.cancel.bits  := 0.U.asTypeOf(coherence.io.cancel.bits)
}
