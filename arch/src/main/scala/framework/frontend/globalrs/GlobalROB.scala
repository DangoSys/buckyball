package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.PostGDCmd
import framework.frontend.scoreboard.BankScoreboard

@instantiable
class GlobalROB(val b: GlobalConfig) extends Module {

  val robDepth  = b.frontend.rob_entries
  val idWidth   = log2Up(robDepth)
  val bankIdLen = b.frontend.bank_id_len

  @public
  val io = IO(new Bundle {
    val alloc    = Flipped(new DecoupledIO(new PostGDCmd(b)))
    val issue    = new DecoupledIO(new GlobalRobEntry(b))
    val complete = Flipped(new DecoupledIO(UInt(idWidth.W)))

    val empty          = Output(Bool())
    val full           = Output(Bool())
    val head_ptr       = Output(UInt(idWidth.W))
    val issued_count   = Output(UInt(log2Up(robDepth + 1).W))
    val entry_valid    = Output(Vec(robDepth, Bool()))
    val entry_complete = Output(Vec(robDepth, Bool()))
  })

  // ---------------------------------------------------------------------------
  // Bank Scoreboard
  // ---------------------------------------------------------------------------
  val scoreboard: Instance[BankScoreboard] = Instantiate(new BankScoreboard(b.memDomain.bankNum, robDepth))

  // ---------------------------------------------------------------------------
  // Instruction trace (DPI-C, defined in ITraceDPI.scala)
  // ---------------------------------------------------------------------------
  val itrace = Module(new ITraceDPI)
  itrace.io.is_issue  := 0.U
  itrace.io.rob_id    := 0.U
  itrace.io.domain_id := 0.U
  itrace.io.funct     := 0.U
  itrace.io.rs1       := 0.U
  itrace.io.rs2       := 0.U
  itrace.io.enable    := false.B

  // ---------------------------------------------------------------------------
  // Storage
  // ---------------------------------------------------------------------------
  val robEntries  = RegInit(VecInit(Seq.fill(robDepth)(0.U.asTypeOf(new GlobalRobEntry(b)))))
  val robValid    = RegInit(VecInit(Seq.fill(robDepth)(false.B)))
  val robIssued   = RegInit(VecInit(Seq.fill(robDepth)(false.B)))
  val robComplete = RegInit(VecInit(Seq.fill(robDepth)(false.B)))

  val headPtr     = RegInit(0.U(idWidth.W))
  val tailPtr     = RegInit(0.U(idWidth.W))
  val issuedCount = RegInit(0.U(log2Up(robDepth + 1).W))

  val isEmpty = headPtr === tailPtr && !robValid(headPtr)
  val isFull  = headPtr === tailPtr && robValid(headPtr)

  def nextPtr(p: UInt): UInt = Mux(p === (robDepth - 1).U, 0.U, p + 1.U)
  def wrapPtr(v: UInt): UInt = Mux(v >= robDepth.U, v - robDepth.U, v)

  // ---------------------------------------------------------------------------
  // Allocate: enqueue decoded instruction into ROB
  // rob_id == tailPtr at allocation time (no separate counter needed)
  // ---------------------------------------------------------------------------
  io.alloc.ready := !isFull

  when(io.alloc.fire) {
    robEntries(tailPtr).cmd    := io.alloc.bits
    robEntries(tailPtr).rob_id := tailPtr
    robValid(tailPtr)          := true.B
    robIssued(tailPtr)         := false.B
    robComplete(tailPtr)       := false.B
    tailPtr                    := nextPtr(tailPtr)
  }

  // ---------------------------------------------------------------------------
  // Complete: mark entry as completed, release scoreboard resources
  // ---------------------------------------------------------------------------
  io.complete.ready := true.B

  scoreboard.complete.valid := false.B
  scoreboard.complete.bits  := 0.U.asTypeOf(scoreboard.complete.bits)

  when(io.complete.fire) {
    val cid = io.complete.bits
    robComplete(cid)          := true.B
    when(robIssued(cid)) {
      issuedCount := issuedCount - 1.U
    }
    scoreboard.complete.valid := true.B
    scoreboard.complete.bits  := robEntries(cid).cmd.bankAccess

    itrace.io.is_issue  := 0.U
    itrace.io.rob_id    := cid
    itrace.io.domain_id := robEntries(cid).cmd.domain_id
    itrace.io.funct     := robEntries(cid).cmd.cmd.funct
    itrace.io.rs1       := robEntries(cid).cmd.cmd.rs1
    itrace.io.rs2       := robEntries(cid).cmd.cmd.rs2
    itrace.io.enable    := true.B
  }

  // ---------------------------------------------------------------------------
  // Issue: scan from head for first issuable entry (valid && !issued && !complete)
  // ---------------------------------------------------------------------------
  val scanValid = Wire(Vec(robDepth, Bool()))
  for (i <- 0 until robDepth) {
    val ptr = wrapPtr(headPtr + i.U)
    scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
  }

  val hasValid       = scanValid.asUInt.orR
  val firstValid     = PriorityEncoder(scanValid.asUInt)
  val actualIssuePtr = wrapPtr(headPtr + firstValid)

  scoreboard.query := robEntries(actualIssuePtr).cmd.bankAccess
  val noHazard = !scoreboard.hasHazard
  val canIssue = hasValid && noHazard

  io.issue.valid := canIssue
  io.issue.bits  := robEntries(actualIssuePtr)

  scoreboard.issue.valid := false.B
  scoreboard.issue.bits  := 0.U.asTypeOf(scoreboard.issue.bits)

  when(io.issue.fire) {
    robIssued(actualIssuePtr) := true.B
    issuedCount               := issuedCount + 1.U
    scoreboard.issue.valid    := true.B
    scoreboard.issue.bits     := robEntries(actualIssuePtr).cmd.bankAccess

    itrace.io.is_issue  := 1.U
    itrace.io.rob_id    := robEntries(actualIssuePtr).rob_id
    itrace.io.domain_id := robEntries(actualIssuePtr).cmd.domain_id
    itrace.io.funct     := robEntries(actualIssuePtr).cmd.cmd.funct
    itrace.io.rs1       := robEntries(actualIssuePtr).cmd.cmd.rs1
    itrace.io.rs2       := robEntries(actualIssuePtr).cmd.cmd.rs2
    itrace.io.enable    := true.B
  }

  // ---------------------------------------------------------------------------
  // Commit: clear completed entries.
  // Explicitly skip entries being allocated or completed this cycle.
  // ---------------------------------------------------------------------------
  for (i <- 0 until robDepth) {
    val beingAllocated = io.alloc.fire && (tailPtr === i.U)
    val beingCompleted = io.complete.fire && (io.complete.bits === i.U)
    when(robValid(i) && robComplete(i) && !beingAllocated && !beingCompleted) {
      robValid(i)    := false.B
      robIssued(i)   := false.B
      robComplete(i) := false.B
    }
  }

  // Update head pointer: advance past all committed entries
  val nextHeadCandidates = Wire(Vec(robDepth, Bool()))
  for (i <- 0 until robDepth) {
    val ptr = wrapPtr(headPtr + i.U)
    nextHeadCandidates(i) := robValid(ptr) && !robComplete(ptr)
  }

  val hasUncommitted = nextHeadCandidates.asUInt.orR
  val nextHeadOffset = PriorityEncoder(nextHeadCandidates.asUInt)
  headPtr := Mux(hasUncommitted, wrapPtr(headPtr + nextHeadOffset), tailPtr)

  // ---------------------------------------------------------------------------
  // Status outputs
  // ---------------------------------------------------------------------------
  io.empty          := isEmpty
  io.full           := isFull
  io.head_ptr       := headPtr
  io.issued_count   := issuedCount
  io.entry_valid    := robValid
  io.entry_complete := robComplete
}
