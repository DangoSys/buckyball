package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.scoreboard.{BankAccessInfo, BankAliasTable, BankScoreboard}
import framework.memdomain.frontend.cmd.decoder.DISA.{MMIO_SET_BITPAT, MSET_BITPAT}

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
  val bankCols    = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(0.U(5.W))))

  val isEmpty = headPtr === tailPtr && !robValid(headPtr)
  val isFull  = headPtr === tailPtr && robValid(headPtr)

  def nextPtr(p: UInt): UInt = Mux(p === (robDepth - 1).U, 0.U, p + 1.U)
  def wrapPtr(v: UInt): UInt = Mux(v >= robDepth.U, v - robDepth.U, v)
  def robIdx(v:  UInt): UInt = v(idWidth - 1, 0)

  def isMappingConfig(entry: GlobalRobEntry): Bool =
    entry.cmd.domain_id === DomainId.MEM &&
      ((entry.cmd.cmd.funct === MSET_BITPAT) || (entry.cmd.cmd.funct === MMIO_SET_BITPAT))

  def touchesBank(access: BankAccessInfo, bank: UInt): Bool =
    (access.rd_bank_0_valid && access.rd_bank_0_id === bank) ||
      (access.rd_bank_1_valid && access.rd_bank_1_id === bank) ||
      (access.wr_bank_valid && access.wr_bank_id === bank)

  def bankOverlap(a: BankAccessInfo, b: BankAccessInfo): Bool =
    (a.rd_bank_0_valid && touchesBank(b, a.rd_bank_0_id)) ||
      (a.rd_bank_1_valid && touchesBank(b, a.rd_bank_1_id)) ||
      (a.wr_bank_valid && touchesBank(b, a.wr_bank_id))

  def accessUsesBank(access: BankAccessInfo, bank: UInt): Bool =
    (access.rd_bank_0_valid && access.rd_bank_0_id === bank) ||
      (access.rd_bank_1_valid && access.rd_bank_1_id === bank) ||
      (access.wr_bank_valid && access.wr_bank_id === bank)

  def hasRawHazard(younger: BankAccessInfo, older: BankAccessInfo): Bool = {
    val youngerReadOlderWrite  =
      older.wr_bank_valid &&
        ((younger.rd_bank_0_valid && younger.rd_bank_0_id === older.wr_bank_id) ||
          (younger.rd_bank_1_valid && younger.rd_bank_1_id === older.wr_bank_id))
    val youngerWriteOlderRead  =
      younger.wr_bank_valid &&
        ((older.rd_bank_0_valid && younger.wr_bank_id === older.rd_bank_0_id) ||
          (older.rd_bank_1_valid && younger.wr_bank_id === older.rd_bank_1_id))
    val youngerWriteOlderWrite =
      younger.wr_bank_valid && older.wr_bank_valid && younger.wr_bank_id === older.wr_bank_id
    youngerReadOlderWrite || youngerWriteOlderRead || youngerWriteOlderWrite
  }

  // ---------------------------------------------------------------------------
  // Allocate: enqueue decoded instruction into ROB
  // rob_id == tailPtr at allocation time (no separate counter needed)
  // ---------------------------------------------------------------------------
  val commitMask = Wire(Vec(robDepth, Bool()))
  for (i <- 0 until robDepth) {
    commitMask(i) := false.B
  }
  bat.io.free.valid := commitMask.asUInt.orR
  bat.io.free.mask := commitMask

  val commitScan = Wire(Vec(robDepth, Bool()))
  val commitKeep = Wire(Vec(robDepth + 1, Bool()))
  commitKeep(0) := true.B
  for (i <- 0 until robDepth) {
    val ptr = robIdx(wrapPtr(headPtr + i.U))
    commitScan(i)     := commitKeep(i) && robValid(ptr) && robComplete(ptr)
    commitKeep(i + 1) := commitScan(i)
  }
  val hasCommit = commitScan.asUInt.orR
  val tailAlias = Wire(UInt(b.frontend.bank_id_len.W))
  tailAlias := (b.frontend.vbank_id_upper_bound + 1).U + tailPtr

  val tailAliasLive = WireDefault(false.B)
  for (i <- 0 until robDepth) {
    when(robValid(i) && !robComplete(i) && accessUsesBank(robEntries(i).renamedBankAccess, tailAlias)) {
      tailAliasLive := true.B
    }
  }

  io.alloc.ready      := !isFull && !hasCommit && !tailAliasLive
  bat.io.alloc.valid  := io.alloc.fire
  bat.io.alloc.rob_id := tailPtr
  bat.io.alloc.raw    := io.alloc.bits.bankAccess

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

  val issueFired          = WireDefault(false.B)
  val completeIssuedEntry = io.complete.fire && robIssued(io.complete.bits)

  when(io.complete.fire) {
    val cid = io.complete.bits
    robComplete(cid)          := true.B
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
    val ptr       = robIdx(wrapPtr(headPtr + i.U))
    val cfgHazard = WireDefault(false.B)
    for (j <- 0 until i) {
      val olderPtr         = robIdx(wrapPtr(headPtr + j.U))
      val olderLive        = robValid(olderPtr) && !robComplete(olderPtr)
      val olderUncommitted = robValid(olderPtr)
      val cfgPair          = isMappingConfig(robEntries(ptr)) || isMappingConfig(robEntries(olderPtr))
      when(olderLive && hasRawHazard(robEntries(ptr).cmd.bankAccess, robEntries(olderPtr).cmd.bankAccess)) {
        cfgHazard := true.B
      }
      when(olderUncommitted && cfgPair && bankOverlap(
        robEntries(ptr).cmd.bankAccess,
        robEntries(olderPtr).cmd.bankAccess
      )) {
        cfgHazard := true.B
      }
    }
    scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
    scoreboard.queryVec(i) := robEntries(ptr).renamedBankAccess
    scanReady(i)           := scanValid(i) && !scoreboard.hazardVec(i) && !cfgHazard
  }

  val hasReady       = scanReady.asUInt.orR
  val firstReady     = PriorityEncoder(scanReady.asUInt)
  val actualIssuePtr = robIdx(wrapPtr(headPtr + firstReady))

  scoreboard.query := robEntries(actualIssuePtr).renamedBankAccess
  val canIssue = hasReady

  val issueEntry = Wire(new GlobalRobEntry(b))
  issueEntry             := robEntries(actualIssuePtr)
  issueEntry.cmd.op1_col := Mux(
    issueEntry.cmd.bankAccess.rd_bank_0_valid,
    bankCols(issueEntry.cmd.bankAccess.rd_bank_0_id(log2Up(b.memDomain.bankNum) - 1, 0)),
    0.U
  )
  issueEntry.cmd.op2_col := Mux(
    issueEntry.cmd.bankAccess.rd_bank_1_valid,
    bankCols(issueEntry.cmd.bankAccess.rd_bank_1_id(log2Up(b.memDomain.bankNum) - 1, 0)),
    0.U
  )
  issueEntry.cmd.wr_col  := Mux(
    issueEntry.cmd.bankAccess.wr_bank_valid,
    bankCols(issueEntry.cmd.bankAccess.wr_bank_id(log2Up(b.memDomain.bankNum) - 1, 0)),
    0.U
  )

  io.issue.valid := canIssue && !io.subRobActive
  io.issue.bits  := issueEntry

  scoreboard.issue.valid := false.B
  scoreboard.issue.bits  := 0.U.asTypeOf(scoreboard.issue.bits)

  when(io.issue.fire) {
    robIssued(actualIssuePtr) := true.B
    issueFired                := true.B
    scoreboard.issue.valid    := true.B
    scoreboard.issue.bits     := robEntries(actualIssuePtr).renamedBankAccess

    itraceIssue.io.is_issue    := 1.U
    itraceIssue.io.rob_id      := issueEntry.rob_id
    itraceIssue.io.domain_id   := issueEntry.cmd.domain_id
    itraceIssue.io.funct       := issueEntry.cmd.cmd.funct
    itraceIssue.io.pc          := issueEntry.cmd.cmd.pc
    itraceIssue.io.rs1         := issueEntry.cmd.cmd.rs1
    itraceIssue.io.rs2         := issueEntry.cmd.cmd.rs2
    itraceIssue.io.bank_enable := issueEntry.cmd.cmd.funct(6, 4)
    itraceIssue.io.enable      := true.B
  }

  issuedCount := issuedCount + issueFired.asUInt - completeIssuedEntry.asUInt

  // ---------------------------------------------------------------------------
  // Commit: clear completed entries.
  // Explicitly skip entries being allocated or completed this cycle.
  // ---------------------------------------------------------------------------
  for (i <- 0 until robDepth) {
    val hits = (0 until robDepth).map { off =>
      val ptr = robIdx(wrapPtr(headPtr + off.U))
      commitScan(off) && ptr === i.U
    }
    commitMask(i) := hits.reduce(_ || _)
    when(commitMask(i)) {
      when(robEntries(i).cmd.domain_id === DomainId.MEM && robEntries(i).cmd.cmd.funct === MSET_BITPAT) {
        val bank = robEntries(i).cmd.bankAccess.wr_bank_id(log2Up(b.memDomain.bankNum) - 1, 0)
        val col  = robEntries(i).cmd.cmd.rs2Data(9, 5)
        when(robEntries(i).cmd.cmd.rs2Data(10)) {
          bankCols(bank) := col
        }.otherwise {
          bankCols(bank) := 0.U
        }
      }
      robValid(i)    := false.B
      robIssued(i)   := false.B
      robComplete(i) := false.B
    }
  }

  val commitCount = PopCount(commitScan)
  headPtr := wrapPtr(headPtr + commitCount)

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
