package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.PostGDCmd
import framework.frontend.scoreboard.{BankAliasTable, BankScoreboard}

@instantiable
class GlobalROB(val b: GlobalConfig) extends Module {

  val robDepth     = b.frontend.rob_entries
  val idWidth      = log2Up(robDepth)
  val scoreBankNum = 1 << b.frontend.bank_id_len

  require(
    b.frontend.vbank_id_upper_bound < b.memDomain.bankNum,
    s"vbank_id_upper_bound(${b.frontend.vbank_id_upper_bound}) must be < memDomain.bankNum(${b.memDomain.bankNum})"
  )

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

    val subRobActive = Input(Bool())
  })

  // ---------------------------------------------------------------------------
  // BAT + Bank Scoreboard
  // ---------------------------------------------------------------------------
  val bat: Instance[BankAliasTable] = Instantiate(
    new BankAliasTable(
      bankIdLen = b.frontend.bank_id_len,
      vbankUpper = b.frontend.vbank_id_upper_bound,
      robEntries = robDepth
    )
  )

  val scoreboard: Instance[BankScoreboard] = Instantiate(new BankScoreboard(scoreBankNum, robDepth))

  // ---------------------------------------------------------------------------
  // Instruction trace (DPI-C, defined in ITraceDPI.scala)
  // ---------------------------------------------------------------------------
  val itraceAlloc = Module(new ITraceDPI)
  val itraceIssue = Module(new ITraceDPI)
  val itraceComp  = Module(new ITraceDPI)

  for (t <- Seq(itraceAlloc, itraceIssue, itraceComp)) {
    t.io.is_issue    := 0.U
    t.io.rob_id      := 0.U
    t.io.domain_id   := 0.U
    t.io.funct       := 0.U
    t.io.pc          := 0.U
    t.io.rs1         := 0.U
    t.io.rs2         := 0.U
    t.io.bank_enable := 0.U
    t.io.enable      := false.B
  }

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
  io.alloc.ready      := !isFull
  bat.io.alloc.valid  := io.alloc.fire
  bat.io.alloc.rob_id := tailPtr
  bat.io.alloc.raw    := io.alloc.bits.bankAccess

  val commitMask = Wire(Vec(robDepth, Bool()))
  for (i <- 0 until robDepth) {
    commitMask(i) := false.B
  }
  bat.io.free.valid := commitMask.asUInt.orR
  bat.io.free.mask := commitMask

  // Mark write alias as busy in scoreboard at alloc time (not issue time).
  scoreboard.alloc.valid := io.alloc.fire && io.alloc.bits.bankAccess.wr_bank_valid
  scoreboard.alloc.bits  := bat.io.alloc_renamed

  when(io.alloc.fire) {
    itraceAlloc.io.is_issue    := 2.U
    itraceAlloc.io.rob_id      := tailPtr
    itraceAlloc.io.domain_id   := io.alloc.bits.domain_id
    itraceAlloc.io.funct       := io.alloc.bits.cmd.funct
    itraceAlloc.io.pc          := io.alloc.bits.cmd.pc
    itraceAlloc.io.rs1         := io.alloc.bits.cmd.rs1
    itraceAlloc.io.rs2         := io.alloc.bits.cmd.rs2
    itraceAlloc.io.bank_enable := io.alloc.bits.cmd.funct(6, 4)
    itraceAlloc.io.enable      := true.B

    robEntries(tailPtr).cmd               := io.alloc.bits
    robEntries(tailPtr).renamedBankAccess := bat.io.alloc_renamed
    robEntries(tailPtr).rob_id            := tailPtr
    robValid(tailPtr)                     := true.B
    robIssued(tailPtr)                    := false.B
    robComplete(tailPtr)                  := false.B
    tailPtr                               := nextPtr(tailPtr)
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
    scoreboard.complete.bits  := robEntries(cid).renamedBankAccess

    itraceComp.io.is_issue    := 0.U
    itraceComp.io.rob_id      := cid
    itraceComp.io.domain_id   := robEntries(cid).cmd.domain_id
    itraceComp.io.funct       := robEntries(cid).cmd.cmd.funct
    itraceComp.io.pc          := robEntries(cid).cmd.cmd.pc
    itraceComp.io.rs1         := robEntries(cid).cmd.cmd.rs1
    itraceComp.io.rs2         := robEntries(cid).cmd.cmd.rs2
    itraceComp.io.bank_enable := robEntries(cid).cmd.cmd.funct(6, 4)
    itraceComp.io.enable      := true.B
  }

  // ---------------------------------------------------------------------------
  // Issue: scan from head for first issuable entry (valid && !issued && !complete)
  // ---------------------------------------------------------------------------
  val scanValid = Wire(Vec(robDepth, Bool()))
  val scanReady = Wire(Vec(robDepth, Bool()))
  for (i <- 0 until robDepth) {
    val ptr = wrapPtr(headPtr + i.U)
    scanValid(i)           := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
    scoreboard.queryVec(i) := robEntries(ptr).renamedBankAccess
    scanReady(i)           := scanValid(i) && !scoreboard.hazardVec(i)
  }

  val hasReady       = scanReady.asUInt.orR
  val firstReady     = PriorityEncoder(scanReady.asUInt)
  val actualIssuePtr = wrapPtr(headPtr + firstReady)

  scoreboard.query := robEntries(actualIssuePtr).renamedBankAccess
  val canIssue = hasReady

  io.issue.valid := canIssue && !io.subRobActive
  io.issue.bits  := robEntries(actualIssuePtr)

  scoreboard.issue.valid := false.B
  scoreboard.issue.bits  := 0.U.asTypeOf(scoreboard.issue.bits)

  when(io.issue.fire) {
    robIssued(actualIssuePtr) := true.B
    issuedCount               := issuedCount + 1.U
    scoreboard.issue.valid    := true.B
    scoreboard.issue.bits     := robEntries(actualIssuePtr).renamedBankAccess

    itraceIssue.io.is_issue    := 1.U
    itraceIssue.io.rob_id      := robEntries(actualIssuePtr).rob_id
    itraceIssue.io.domain_id   := robEntries(actualIssuePtr).cmd.domain_id
    itraceIssue.io.funct       := robEntries(actualIssuePtr).cmd.cmd.funct
    itraceIssue.io.pc          := robEntries(actualIssuePtr).cmd.cmd.pc
    itraceIssue.io.rs1         := robEntries(actualIssuePtr).cmd.cmd.rs1
    itraceIssue.io.rs2         := robEntries(actualIssuePtr).cmd.cmd.rs2
    itraceIssue.io.bank_enable := robEntries(actualIssuePtr).cmd.cmd.funct(6, 4)
    itraceIssue.io.enable      := true.B
  }

  // ---------------------------------------------------------------------------
  // Commit: clear completed entries.
  // Explicitly skip entries being allocated or completed this cycle.
  // ---------------------------------------------------------------------------
  for (i <- 0 until robDepth) {
    val beingAllocated = io.alloc.fire && (tailPtr === i.U)
    val beingCompleted = io.complete.fire && (io.complete.bits === i.U)
    commitMask(i) := robValid(i) && robComplete(i) && !beingAllocated && !beingCompleted
    when(commitMask(i)) {
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
