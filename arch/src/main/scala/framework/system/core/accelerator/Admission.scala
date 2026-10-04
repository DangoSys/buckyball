package framework.system.core.accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import framework.top.GlobalConfig
import framework.system.core.rocket.{RoCCCommandBB, RoCCIO, RoCCResponseBB}
import framework.frontend.decoder.GISA
import framework.memdomain.frontend.cmd.decoder.DISA
import framework.memdomain.frontend.mem.Footprint
import framework.memdomain.frontend.mem.dma.{Dma, DmaDecision, DmaError, DmaPort, DmaStatus}
import hier.core.rocket.{AdmissionBridge, AdmissionPorts, Preparation, RobFault}
import hier.tile.TaskAdmission
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.interlock.{AccessInfo, Interlock, Params => TrackingParams}
import memcore.memory.preflight.{Command, Error, Params => PreparationParams}
import memcore.bus.axi4

/** Per-compute-core admission, physical preparation and completion join. */
@instantiable
class Admission(
  b:                      GlobalConfig,
  tracking:               TrackingParams,
  prepared:               PreparationParams,
  regions:                Seq[PhysicalRegion],
  axiParams:              axi4.Params
)(
  implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  require(tracking.entries == 4 && tracking.idBits == prepared.idBits)
  require(tracking.addressBits == prepared.bus.addressBits && tracking.maxRanges == prepared.maxRanges)
  require(prepared.beatBytes == axiParams.bytes && b.memDomain.dma_buswidth == axiParams.dataBits)
  private val n         = tracking.entries
  private val indexBits = log2Ceil(n)

  @public val io = IO(new Bundle {
    val core         = Flipped(new AdmissionPorts(tracking, nPMPs, prepared.bus))
    val npuCommand   = Decoupled(new RoCCCommandBB)
    val npuResponse  = Flipped(Decoupled(new RoCCResponseBB))
    val allocation   = Flipped(Valid(UInt(log2Ceil(b.frontend.rob_entries).W)))
    val retired      = Input(UInt(b.frontend.rob_entries.W))
    val npuFault     = Flipped(Valid(new RobFault(b.frontend.rob_entries)))
    // Actual scheduler/boot work, not merely RoCC ROB backpressure.
    val npuBusy      = Input(Bool())
    val npuInterrupt = Input(Bool())
    val footprints   = Input(Vec(3, new Footprint(b)))
    val dma          = Flipped(new DmaPort(b.memDomain.dma_buswidth))
    val task         = Flipped(new RoCCIO(64))
    val taskSatp     = Output(UInt(64.W))
    val axi          = new axi4.Port(axiParams)
    val halted       = Output(Bool())
    val fault        = Output(new DmaStatus)
    val faultTag     = Output(UInt(tracking.idBits.W))
    val faultHasTag  = Output(Bool())
    val workDrained  = Output(Bool())
  })

  val bridge = Instantiate(new AdmissionBridge(
    b.frontend.rob_entries,
    tracking,
    nPMPs,
    Seq(GISA.FENCE_BITPAT.value.toInt, GISA.BARRIER_BITPAT.value.toInt)
  ))

  val interlock   = Instantiate(new Interlock(tracking))
  val preparation = Instantiate(new Preparation(prepared, regions))
  val dma         = Instantiate(new Dma(b, prepared, axiParams))
  val task        = Instantiate(new TaskAdmission(tracking, nPMPs))

  val live            = RegInit(VecInit(Seq.fill(n)(false.B)))
  val memory          = RegInit(VecInit(Seq.fill(n)(false.B)))
  val fence           = RegInit(VecInit(Seq.fill(n)(false.B)))
  val retirementSeen  = RegInit(VecInit(Seq.fill(n)(false.B)))
  val interlockDone   = RegInit(VecInit(Seq.fill(n)(false.B)))
  val noMemoryPending = RegInit(VecInit(Seq.fill(n)(false.B)))
  val shapeClaimed    = RegInit(VecInit(Seq.fill(n)(false.B)))
  val preparedStarted = RegInit(VecInit(Seq.fill(n)(false.B)))
  val mapReady        = RegInit(VecInit(Seq.fill(n)(false.B)))
  val grantSeen       = RegInit(VecInit(Seq.fill(n)(false.B)))
  val doneIssued      = RegInit(VecInit(Seq.fill(n)(false.B)))
  val errors          = RegInit(VecInit(Seq.fill(n)(0.U.asTypeOf(new DmaStatus))))
  val halted          = RegInit(false.B)
  val firstFault      = RegInit(0.U.asTypeOf(new DmaStatus))
  val firstTag        = RegInit(0.U(tracking.idBits.W))
  val firstHasTag     = RegInit(false.B)
  def index(tag: UInt): UInt = tag(indexBits - 1, 0)
  def known(tag: UInt): Bool = tag < n.U && live(index(tag))

  def stop(
    error:   UInt,
    address: UInt,
    tag:     UInt,
    hasTag:  Bool
  ): Unit = {
    when(!halted) {
      halted   := true.B; firstFault.error := error; firstFault.address := address
      firstTag := tag; firstHasTag         := hasTag
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

  val command = io.core.command
  val isTask  = command.bits.instruction.opcode === "h2b".U
  val isFence = !isTask && (command.bits.instruction.funct === GISA.FENCE_BITPAT ||
    command.bits.instruction.funct === GISA.BARRIER_BITPAT)

  val isMemory = !isTask && (command.bits.instruction.funct === DISA.MVIN_BITPAT ||
    command.bits.instruction.funct === DISA.MVIN_2D_BITPAT ||
    command.bits.instruction.funct === DISA.MVIN_MMIO_BITPAT ||
    command.bits.instruction.funct === DISA.MVOUT_BITPAT ||
    command.bits.instruction.funct === GISA.MVIN_KERNEL_BITPAT)

  val fenceProtect = fence.asUInt.orR || (command.valid && isFence)
  interlock.io.dispatch.valid          := io.core.reserve.valid && !halted && !fenceProtect
  interlock.io.dispatch.bits           := io.core.reserve.bits
  io.core.reserve.ready                := interlock.io.dispatch.ready && !halted && !fenceProtect
  when(io.core.reserve.fire) {
    val tag = io.core.reserve.bits.id
    assert(tag < n.U && !live(index(tag)), "Admission reused a live Core tag")
    live(index(tag))            := true.B; memory(index(tag))         := false.B; fence(index(tag))           := false.B
    retirementSeen(index(tag))  := false.B; interlockDone(index(tag)) := false.B
    noMemoryPending(index(tag)) := false.B; shapeClaimed(index(tag))  := false.B; preparedStarted(index(tag)) := false.B
    mapReady(index(tag))        := false.B; grantSeen(index(tag))     := false.B; doneIssued(index(tag))      := false.B
    errors(index(tag))          := 0.U.asTypeOf(new DmaStatus)
  }
  bridge.io.command.valid              := command.valid && !isTask && !halted
  bridge.io.command.bits               := command.bits
  task.io.command.valid                := command.valid && isTask && !halted
  task.io.command.bits                 := command.bits
  command.ready                        := !halted && Mux(isTask, task.io.command.ready, bridge.io.command.ready)
  when(command.fire) {
    assert(known(command.bits.tag), "Admission command has no reserved Core tag")
    val slot = index(command.bits.tag)
    memory(slot)          := isMemory; fence(slot) := isFence
    noMemoryPending(slot) := !isMemory
  }
  io.npuCommand <> bridge.io.npuCommand
  bridge.io.allocation                 := io.allocation; bridge.io.retired := io.retired; bridge.io.fault := io.npuFault
  io.task <> task.io.task; io.taskSatp := task.io.satp
  val responses = Module(new Arbiter(new RoCCResponseBB, 2))
  responses.io.in(0) <> task.io.response; responses.io.in(1) <> io.npuResponse
  io.core.response <> responses.io.out
  io.core.interrupt         := io.npuInterrupt || io.task.interrupt
  interlock.io.cpuQuery     := io.core.cpuQuery
  io.core.cpuAllow          := interlock.io.cpuAllow && !halted && !fenceProtect
  io.core.cpuProbeAllow     := interlock.io.cpuProbeAllow && !halted && !fenceProtect
  io.core.maintenance <> interlock.io.maintenance
  interlock.io.maintained <> io.core.maintained
  io.core.pteRequest <> preparation.io.pteRequest
  preparation.io.pteResponse <> io.core.pteResponse
  interlock.io.cancel.valid := false.B; interlock.io.cancel.bits := 0.U.asTypeOf(interlock.io.cancel.bits)
  io.core.cancelled.valid   := false.B; io.core.cancelled.bits   := 0.U.asTypeOf(io.core.cancelled.bits)

  // Notices never wait for other tags: otherwise a high-priority fence notice
  // could prevent older ROB retirements from reaching the completion join.
  bridge.io.retirement.ready  := true.B
  when(bridge.io.retirement.fire) {
    val notice = bridge.io.retirement.bits
    assert(known(notice.snapshot.tag), "Admission retirement has no live Core tag")
    retirementSeen(index(notice.snapshot.tag)) := true.B
    when(notice.error =/= DmaError.None.U) {
      errors(index(notice.snapshot.tag)).error   := notice.error
      errors(index(notice.snapshot.tag)).address := notice.address
      stop(notice.error, notice.address, notice.snapshot.tag, true.B)
    }
  }
  task.io.release.ready       := true.B
  when(task.io.release.fire) {
    assert(known(task.io.release.bits.tag), "Task release has no live Core tag")
    retirementSeen(index(task.io.release.bits.tag)) := true.B
  }
  interlock.io.complete.ready := true.B
  when(interlock.io.complete.fire) {
    assert(known(interlock.io.complete.bits.tag), "Interlock complete has no live Core tag")
    interlockDone(index(interlock.io.complete.bits.tag)) := true.B
  }
  when(bridge.io.unboundFault.valid) {
    stop(bridge.io.unboundFault.bits.error, bridge.io.unboundFault.bits.address, 0.U, false.B)
  }

  val producerBound = RegInit(VecInit(Seq.fill(3)(false.B)))
  val producerTag   = Reg(Vec(3, UInt(tracking.idBits.W)))
  val shapes        = Reg(Vec(3, new Command(prepared)))
  val shapePending  = RegInit(VecInit(Seq.fill(3)(false.B)))
  for (i <- 0 until 3) {
    bridge.io.lookup(i).robId := io.footprints(i).rob_id
    val footprint = io.footprints(i)
    val snapshot  = bridge.io.lookup(i).snapshot
    when(producerBound(i) && !footprint.valid)(producerBound(i) := false.B)
    when(footprint.valid && !producerBound(i) && !halted && !io.npuFault.valid) {
      when(!snapshot.valid || !known(snapshot.bits.tag) || !memory(index(snapshot.bits.tag)) || footprint.is_sub) {
        stop(DmaError.Context.U, footprint.baseVA, snapshot.bits.tag, snapshot.valid)
      }.otherwise {
        val tag    = snapshot.bits.tag; val slot = index(tag)
        val narrow = footprint.rows(63, 32).orR || footprint.columns(63, 32).orR ||
          footprint.spanBytes(63, 32).orR || footprint.columnStride(63, 32).orR || footprint.rowStride(63, 32).orR
        producerBound(i)    := true.B; producerTag(i)                             := tag
        shapes(i).id        := tag; shapes(i).baseVA                              := footprint.baseVA
        shapes(i).rows      := footprint.rows(31, 0); shapes(i).columns           := footprint.columns(31, 0)
        shapes(i).spanBytes := footprint.spanBytes(31, 0); shapes(i).columnStride := footprint.columnStride(31, 0)
        shapes(i).rowStride := footprint.rowStride(31, 0); shapes(i).write        := footprint.write
        shapes(i).mode      := snapshot.bits.satp(63, 60); shapes(i).rootPpn      := snapshot.bits.satp(43, 0)
        shapes(i).privilege := snapshot.bits.effectivePrivilege
        shapes(i).sum       := snapshot.bits.sum; shapes(i).mxr                   := snapshot.bits.mxr
        when(narrow || footprint.fault.error =/= DmaError.None.U) {
          val error = Mux(narrow, DmaError.Shape.U, footprint.fault.error)
          errors(slot).error   := error
          errors(slot).address := Mux(narrow, footprint.baseVA, footprint.fault.address)
          stop(error, Mux(narrow, footprint.baseVA, footprint.fault.address), tag, true.B)
        }.otherwise {
          when(shapeClaimed(slot)) {
            errors(slot).error := DmaError.Context.U; errors(slot).address := footprint.baseVA
            stop(DmaError.Context.U, footprint.baseVA, tag, true.B)
          }.otherwise { shapeClaimed(slot) := true.B; shapePending(i) := true.B }
        }
      }
    }
  }
  val prepareOffer = RegInit(false.B)
  val prepareProducer = Reg(UInt(2.W))
  when(!prepareOffer && shapePending.asUInt.orR && !halted) {
    prepareOffer := true.B; prepareProducer := PriorityEncoder(shapePending)
  }
  preparation.io.command.valid := prepareOffer && !halted
  preparation.io.command.bits := shapes(prepareProducer)
  when(preparation.io.command.fire) {
    prepareOffer                                       := false.B; shapePending(prepareProducer) := false.B
    preparedStarted(index(shapes(prepareProducer).id)) := true.B
  }
  bridge.io.lookupTag.tag     := preparation.io.contextTag
  preparation.io.contextValid := bridge.io.lookupTag.snapshot.valid
  preparation.io.contextId    := bridge.io.lookupTag.snapshot.bits.tag
  preparation.io.contextPmp   := bridge.io.lookupTag.snapshot.bits.pmp

  val infos      = Module(new Queue(new AccessInfo(tracking), 2))
  interlock.io.accessInfo <> infos.io.deq
  val range      = preparation.io.ranges
  val infoOffer  = RegInit(false.B)
  val infoRange  = RegInit(false.B)
  val infoPacket = Reg(new AccessInfo(tracking))
  val pure       = PriorityEncoder(noMemoryPending)
  when(!infoOffer && !halted &&
    ((range.valid && range.bits.error === Error.Ok.U) || (!range.valid && noMemoryPending.asUInt.orR))) {
    infoOffer            := true.B; infoRange := range.valid && range.bits.error === Error.Ok.U
    infoPacket           := 0.U.asTypeOf(infoPacket)
    infoPacket.id        := Mux(range.valid, range.bits.id, pure)
    infoPacket.hasMemory := range.valid
    infoPacket.base      := Mux(range.valid, range.bits.pa, 0.U)
    infoPacket.bytes     := Mux(range.valid, range.bits.bytes, 0.U)
    infoPacket.write     := range.valid && range.bits.write
    infoPacket.last      := Mux(range.valid, range.bits.last, true.B)
  }
  infos.io.enq.valid := infoOffer && !halted
  infos.io.enq.bits                                                           := infoPacket
  range.ready                                                                 := Mux(range.bits.error =/= Error.Ok.U, true.B, infoOffer && infoRange && infos.io.enq.ready && !halted)
  when(infos.io.enq.fire) {
    infoOffer                                              := false.B
    when(!infoRange)(noMemoryPending(index(infoPacket.id)) := false.B)
  }
  when(range.fire && range.bits.error =/= Error.Ok.U) {
    val error = dmaError(range.bits.error)
    errors(index(range.bits.id)).error := error; errors(index(range.bits.id)).address := range.bits.va
    stop(error, range.bits.va, range.bits.id, true.B)
  }
  preparation.io.ready.ready                                                  := true.B
  when(preparation.io.ready.fire) {
    val value = preparation.io.ready.bits
    mapReady(index(value.id)) := true.B
    when(value.error =/= Error.Ok.U) {
      errors(index(value.id)).error := dmaError(value.error); errors(index(value.id)).address := value.va
      stop(dmaError(value.error), value.va, value.id, true.B)
    }
  }
  interlock.io.grant.ready                                                    := true.B
  when(interlock.io.grant.fire)(grantSeen(index(interlock.io.grant.bits.tag)) := true.B)

  preparation.io.queries := dma.io.queries
  for (i <- 0 until 2) {
    dma.io.mappings(i) := preparation.io.results(i)
    when(halted) {
      dma.io.mappings(i).hit   := false.B; dma.io.mappings(i).pa := 0.U
      dma.io.mappings(i).error := Error.Context.U
    }
  }
  when(io.npuFault.valid && io.npuFault.bits.error =/= DmaError.None.U) {
    stop(io.npuFault.bits.error, io.npuFault.bits.address, bridge.io.faultTag.bits, bridge.io.faultTag.valid)
  }
  val readOwnerTag = Reg(UInt(tracking.idBits.W))
  val writeOwnerTag = Reg(UInt(tracking.idBits.W))
  when(io.dma.read.fire)(readOwnerTag   := Mux(io.dma.read.bits.producer === 2.U, producerTag(2), producerTag(0)))
  when(io.dma.write.fire)(writeOwnerTag := producerTag(1))
  dma.io.dma <> io.dma
  io.axi <> dma.io.axi
  for (i <- 0 until 3) {
    val tag = producerTag(i); val slot = index(tag)
    dma.io.decisions(i)          := 0.U.asTypeOf(new DmaDecision(prepared))
    dma.io.decisions(i).parentId := tag
    dma.io.decisions(i).valid    := producerBound(i) &&
      ((mapReady(slot) && grantSeen(slot)) || errors(slot).error =/= DmaError.None.U || halted)
    dma.io.decisions(i).fault    := Mux(halted, firstFault, errors(slot))
  }
  when(io.dma.readResult.fire && io.dma.readResult.bits.fault.error =/= DmaError.None.U) {
    stop(io.dma.readResult.bits.fault.error, io.dma.readResult.bits.fault.address, readOwnerTag, true.B)
  }
  when(io.dma.writeResult.fire && io.dma.writeResult.bits.fault.error =/= DmaError.None.U) {
    stop(io.dma.writeResult.bits.fault.error, io.dma.writeResult.bits.fault.address, writeOwnerTag, true.B)
  }

  val doneCandidates = VecInit((0 until n).map(i =>
    live(i) && memory(i) && retirementSeen(i) &&
      mapReady(i) && grantSeen(i) && !doneIssued(i) && errors(i).error === DmaError.None.U
  ))

  val doneOffer = RegInit(false.B)
  val doneTag   = Reg(UInt(indexBits.W))
  when(!doneOffer && doneCandidates.asUInt.orR && !halted) {
    doneOffer := true.B; doneTag := PriorityEncoder(doneCandidates)
  }
  interlock.io.done.valid := doneOffer && !halted
  interlock.io.done.bits.tag := doneTag; interlock.io.done.bits.ok := true.B
  when(interlock.io.done.fire) { doneIssued(doneTag) := true.B; doneOffer := false.B }

  val releasable = VecInit((0 until n).map(i =>
    live(i) && retirementSeen(i) && interlockDone(i) &&
      errors(i).error === DmaError.None.U && (!fence(i) ||
        (PopCount(live) === 1.U && !io.npuBusy && !io.dma.readBusy && !io.dma.writeBusy))
  ))

  val releaseOffer = RegInit(false.B)
  val releaseTag   = Reg(UInt(indexBits.W))
  when(!releaseOffer && releasable.asUInt.orR && !halted) {
    releaseOffer := true.B; releaseTag := PriorityEncoder(releasable)
  }
  val releasingMap = memory(releaseTag) && preparedStarted(releaseTag)
  preparation.io.release.valid   := releaseOffer && releasingMap && io.core.complete.ready && !halted
  preparation.io.release.bits.id := releaseTag
  io.core.complete.valid         := releaseOffer && !halted && (!releasingMap || preparation.io.release.ready)
  io.core.complete.bits.tag      := releaseTag
  when(io.core.complete.fire) {
    assert(!releasingMap || preparation.io.release.fire, "Admission map/Core release is not atomic")
    live(releaseTag) := false.B; fence(releaseTag) := false.B; releaseOffer := false.B
  }
  val otherLive = (live.asUInt & ~(1.U(n.W) << index(command.bits.tag))).orR

  val drained = !otherLive && !io.npuBusy && !io.dma.readBusy && !io.dma.writeBusy &&
    !prepareOffer && !shapePending.asUInt.orR &&
    !interlock.io.maintenance.valid && !halted

  task.io.workDrained := drained
  io.workDrained      := !live.asUInt.orR && !io.npuBusy && !io.dma.readBusy && !io.dma.writeBusy &&
    !halted
  io.halted           := halted; io.fault := firstFault; io.faultTag := firstTag; io.faultHasTag := firstHasTag
}
