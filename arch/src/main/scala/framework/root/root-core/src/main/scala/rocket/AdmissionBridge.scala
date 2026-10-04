package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import framework.system.core.rocket.RoCCCommandBB
import memcore.memory.interlock.{Params => TrackingParams}

class RobFault(robEntries: Int) extends Bundle {
  val rob_id  = UInt(log2Ceil(robEntries).W)
  val error   = UInt(4.W)
  val address = UInt(64.W)
}

class AdmissionRetirement(tracking: TrackingParams, pmps: Int)(implicit val cpuParams: CpuParams)
    extends Bundle
    with HasCpuParameters {
  val snapshot = new CommandSnapshot(tracking, pmps)
  val error    = UInt(4.W)
  val address  = UInt(64.W)
}

/** Associates accepted CPU snapshots with actual external-command ROB allocations. */
@instantiable
class AdmissionBridge(
  robEntries:             Int,
  tracking:               TrackingParams,
  pmps:                   Int,
  nonRobFuncts:           Seq[Int]
)(
  implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  require(robEntries >= 2 && tracking.entries == 4)
  private val robBits = log2Ceil(robEntries)

  @public val io = IO(new Bundle {
    val command      = Flipped(Decoupled(new CommandSnapshot(tracking, pmps)))
    val npuCommand   = Decoupled(new RoCCCommandBB)
    // bits is the current tail preview even when valid is low; valid means exact alloc.fire.
    val allocation   = Flipped(Valid(UInt(robBits.W)))
    val retired      = Input(UInt(robEntries.W))
    val fault        = Flipped(Valid(new RobFault(robEntries)))
    val unboundFault = Valid(new RobFault(robEntries))
    val faultTag     = Valid(UInt(tracking.idBits.W))

    val lookup = Vec(
      3,
      new Bundle {
        val robId    = Input(UInt(robBits.W))
        val snapshot = Output(Valid(new CommandSnapshot(tracking, pmps)))
      }
    )

    val lookupTag = new Bundle {
      val tag      = Input(UInt(tracking.idBits.W))
      val snapshot = Output(Valid(new CommandSnapshot(tracking, pmps)))
    }

    // A ROB retirement notice alone does not establish DMA/cache completion.
    val retirement = Decoupled(new AdmissionRetirement(tracking, pmps))
  })

  val live         = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
  val pending      = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
  val hasRob       = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
  val robIds       = Reg(Vec(tracking.entries, UInt(robBits.W)))
  val snapshots    = Reg(Vec(tracking.entries, new CommandSnapshot(tracking, pmps)))
  val errors       = RegInit(VecInit(Seq.fill(tracking.entries)(0.U(4.W))))
  val addresses    = RegInit(VecInit(Seq.fill(tracking.entries)(0.U(64.W))))
  val free         = VecInit(live.map(!_)).asUInt
  val slot         = PriorityEncoder(free)
  val nonRob       = nonRobFuncts.map(f => io.command.bits.instruction.funct === f.U(7.W)).reduce(_ || _)
  val duplicateTag =
    (0 until tracking.entries).map(i => live(i) && snapshots(i).tag === io.command.bits.tag).reduce(_ || _)
  val duplicateRob =
    (0 until tracking.entries).map(i => live(i) && hasRob(i) && robIds(i) === io.allocation.bits).reduce(_ || _)
  val eligible     = free.orR && !duplicateTag && (nonRob || !duplicateRob) && !reset.asBool
  io.npuCommand.valid := io.command.valid && eligible
  io.npuCommand.bits  := io.command.bits.instruction
  io.command.ready    := io.npuCommand.ready && eligible

  when(io.npuCommand.fire) {
    assert(io.allocation.valid === !nonRob, "AdmissionBridge command/ROB allocation mismatch")
    when(!nonRob)(assert(io.allocation.bits < robEntries.U, "AdmissionBridge ROB allocation out of range"))
    live(slot)      := true.B
    pending(slot)   := nonRob
    hasRob(slot)    := !nonRob
    robIds(slot)    := io.allocation.bits
    snapshots(slot) := io.command.bits
    errors(slot)    := 0.U
    addresses(slot) := 0.U
  }
  // Boot is reset-only in the actual Frontend: its allocations precede all external contexts.
  when(io.allocation.valid && !io.npuCommand.fire) {
    assert(!live.asUInt.orR, "AdmissionBridge boot allocation overlaps a live CPU context")
  }
  val heldCommand     = RegNext(io.npuCommand.valid && !io.npuCommand.ready, false.B)
  val previousCommand = RegEnable(io.npuCommand.bits.asUInt, io.npuCommand.valid)
  when(heldCommand && !reset.asBool) {
    assert(
      io.npuCommand.valid && io.npuCommand.bits.asUInt === previousCommand,
      "AdmissionBridge stalled NPU command changed"
    )
  }
  // Boot-only allocations have no external command fire and create no CPU context.
  for (i    <- 0 until tracking.entries) {
    when(live(i) && hasRob(i) && io.retired(robIds(i)))(pending(i) := true.B)
  }
  for (port <- io.lookup) {
    val hits = VecInit((0 until tracking.entries).map(i =>
      live(i) && hasRob(i) && !pending(i) &&
        !io.retired(robIds(i)) && robIds(i) === port.robId
    ))
    port.snapshot.valid := hits.asUInt.orR && !reset.asBool
    port.snapshot.bits  := Mux1H(hits, snapshots)
    assert(PopCount(hits) <= 1.U, "AdmissionBridge duplicate live ROB binding")
  }

  val tagHits = VecInit((0 until tracking.entries).map(i =>
    live(i) && hasRob(i) && !pending(i) && !io.retired(robIds(i)) && snapshots(i).tag === io.lookupTag.tag
  ))

  io.lookupTag.snapshot.valid := tagHits.asUInt.orR && !reset.asBool
  io.lookupTag.snapshot.bits  := Mux1H(tagHits, snapshots)
  assert(PopCount(tagHits) <= 1.U, "AdmissionBridge duplicate live admission tag")

  val noticeValid = RegInit(false.B)
  val noticeSlot  = RegInit(0.U(log2Ceil(tracking.entries).W))
  when(!noticeValid && pending.asUInt.orR) {
    noticeValid := true.B
    noticeSlot  := PriorityEncoder(pending.asUInt)
  }
  io.retirement.valid := noticeValid && !reset.asBool
  val faultHits   =
    VecInit((0 until tracking.entries).map(i => live(i) && hasRob(i) && robIds(i) === io.fault.bits.rob_id))
  io.unboundFault.valid       := io.fault.valid && !faultHits.asUInt.orR && !reset.asBool
  io.unboundFault.bits        := io.fault.bits
  io.faultTag.valid           := io.fault.valid && faultHits.asUInt.orR && !reset.asBool
  io.faultTag.bits            := Mux1H(faultHits, snapshots.map(_.tag))
  when(io.fault.valid && io.fault.bits.error =/= 0.U) {
    for (i <- 0 until tracking.entries) {
      when(faultHits(i)) {
        val alreadyOffered = noticeValid && noticeSlot === i.U
        assert(!alreadyOffered, "AdmissionBridge fault arrived after retirement offer")
        when(!alreadyOffered && errors(i) === 0.U) {
          errors(i)    := io.fault.bits.error
          addresses(i) := io.fault.bits.address
        }
      }
    }
  }
  io.retirement.bits.snapshot := snapshots(noticeSlot)
  io.retirement.bits.error    := errors(noticeSlot)
  io.retirement.bits.address  := addresses(noticeSlot)
  when(io.retirement.fire) {
    live(noticeSlot)    := false.B
    pending(noticeSlot) := false.B
    hasRob(noticeSlot)  := false.B
    noticeValid         := false.B
  }
}
