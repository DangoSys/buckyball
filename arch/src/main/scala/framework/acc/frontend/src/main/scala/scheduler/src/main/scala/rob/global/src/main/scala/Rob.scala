package framework.acc.frontend.scheduler.rob.global

import chisel3._
import framework.acc.frontend.scheduler.rob.global.configs.RobParams
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class Rob(val p: RobParams) extends Module {

  @public
  val io = IO(new Bundle {

    val alloc = Flipped(Decoupled(new Bundle {
      val data          = UInt(p.dataBits.W)
      val dependencies  = UInt(p.entries.W)
      // Dependencies involving this entry clear only at retirement.
      val commitBarrier = Bool()
    }))

    val issue            = Flipped(Valid(UInt(p.idBits.W)))
    val complete         = Flipped(Valid(UInt(p.idBits.W)))
    val data             = Output(Vec(p.entries, UInt(p.dataBits.W)))
    val valid            = Output(Vec(p.entries, Bool()))
    val issued           = Output(Vec(p.entries, Bool()))
    val completed        = Output(Vec(p.entries, Bool()))
    val dependencies     = Output(Vec(p.entries, UInt(p.entries.W)))
    val retireReady      = Input(Vec(p.entries, Bool()))
    val retireCandidates = Output(UInt(p.entries.W))
    val retired          = Output(UInt(p.entries.W))
    val head             = Output(UInt(p.idBits.W))
    val tail             = Output(UInt(p.idBits.W))
    val issuedCount      = Output(UInt(log2Ceil(p.entries + 1).W))
    val empty            = Output(Bool())
    val full             = Output(Bool())
  })

  val data         = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.dataBits.W))))
  val valid        = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val issued       = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val completed    = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val dependencies = RegInit(VecInit(Seq.fill(p.entries)(0.U(p.entries.W))))
  val barriers     = RegInit(VecInit(Seq.fill(p.entries)(false.B)))
  val head         = RegInit(0.U(p.idBits.W))
  val tail         = RegInit(0.U(p.idBits.W))
  val issuedCount  = RegInit(0.U(log2Ceil(p.entries + 1).W))
  def wrap(value: UInt): UInt = Mux(value >= p.entries.U, value - p.entries.U, value)(p.idBits - 1, 0)

  val candidate = Wire(Vec(p.retireWidth, Bool()))
  val eligible  = Wire(Vec(p.retireWidth + 1, Bool()))
  val retire    = Wire(Vec(p.retireWidth, Bool()))
  val keep      = Wire(Vec(p.retireWidth + 1, Bool()))
  eligible(0) := true.B
  keep(0)     := true.B
  for (offset <- 0 until p.retireWidth) {
    val index = wrap(head +& offset.U)
    candidate(offset)    := eligible(offset) && valid(index) && completed(index)
    eligible(offset + 1) := candidate(offset)
    retire(offset)       := keep(offset) && valid(index) && completed(index) && io.retireReady(index)
    keep(offset + 1)     := retire(offset)
  }
  io.retireCandidates := VecInit((0 until p.entries).map { slot =>
    (0 until p.retireWidth).map(offset => candidate(offset) && wrap(head +& offset.U) === slot.U).reduce(_ || _)
  }).asUInt

  val retired = VecInit((0 until p.entries).map { slot =>
    (0 until p.retireWidth).map(offset => retire(offset) && wrap(head +& offset.U) === slot.U).reduce(_ || _)
  }).asUInt

  val completeMask = Mux(io.complete.valid, UIntToOH(io.complete.bits, p.entries), 0.U(p.entries.W))
  io.empty       := head === tail && !valid(head)
  io.full        := head === tail && valid(head)
  // Preserve the integrated ROB's commit-cycle admission rule.
  io.alloc.ready := !io.full && !retired.orR

  when(io.alloc.fire) {
    data(tail)      := io.alloc.bits.data
    valid(tail)     := true.B
    issued(tail)    := false.B
    completed(tail) := false.B
    barriers(tail)  := io.alloc.bits.commitBarrier
    tail            := wrap(tail +& 1.U)
  }
  when(io.issue.valid) {
    assert(
      io.issue.bits < p.entries.U && valid(io.issue.bits) && !issued(io.issue.bits) && !completed(io.issue.bits),
      "ROB issue must name a live unissued entry"
    )
    issued(io.issue.bits) := true.B
  }
  when(io.complete.valid) {
    assert(
      io.complete.bits < p.entries.U && valid(io.complete.bits) && !completed(io.complete.bits),
      "ROB completion must name a live incomplete entry"
    )
    completed(io.complete.bits) := true.B
  }
  val completingIssued = io.complete.valid &&
    (issued(io.complete.bits) || (io.issue.valid && io.issue.bits === io.complete.bits))
  issuedCount := issuedCount + io.issue.valid.asUInt - completingIssued.asUInt
  for (slot <- 0 until p.entries) {
    when(retired(slot)) {
      valid(slot)     := false.B
      issued(slot)    := false.B
      completed(slot) := false.B
    }
    val allocating = io.alloc.fire && tail === slot.U
    val barrier    = Mux(allocating, io.alloc.bits.commitBarrier, barriers(slot))
    val sticky     = barriers.asUInt | Fill(p.entries, barrier)
    val clear      = retired | (completeMask & ~sticky)
    dependencies(slot) := Mux(allocating, io.alloc.bits.dependencies, dependencies(slot)) & ~clear
  }
  head := wrap(head +& PopCount(retire))
  io.data         := data
  io.valid        := valid
  io.issued       := issued
  io.completed    := completed
  io.dependencies := dependencies
  io.retired      := retired
  io.head         := head
  io.tail         := tail
  io.issuedCount  := issuedCount
}
