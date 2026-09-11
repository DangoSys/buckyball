package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.scoreboard.{BankAccessInfo, BankAliasTable, BankScoreboard}
import framework.memdomain.frontend.cmd.decoder.DISA.MSET_BITPAT

@instantiable
class GlobalROB(val b: GlobalConfig) extends Module {

  val robDepth     = b.frontend.rob_entries
  val idWidth      = log2Up(robDepth)
  val scoreBankNum = b.memDomain.virtualBankCount + robDepth

  require(
    b.frontend.vbank_id_upper_bound < b.memDomain.virtualBankCount,
    s"vbank_id_upper_bound(${b.frontend.vbank_id_upper_bound}) must be < virtualBankCount(${b.memDomain.virtualBankCount})"
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
      vbankUpper = b.memDomain.virtualBankCount - 1,
      robEntries = robDepth
    )
  )

  val scoreboard: Instance[BankScoreboard] =
    Instantiate(new BankScoreboard(b.frontend.bank_id_len, scoreBankNum, robDepth))

  // ---------------------------------------------------------------------------
  // Instruction trace (DPI-C, defined in ITraceDPI.scala)
  // ---------------------------------------------------------------------------
  val itraceAlloc = Module(new ITraceDPI)
  val itraceIssue = Module(new ITraceDPI)
  val itraceComp  = Module(new ITraceDPI)

  for (t <- Seq(itraceAlloc, itraceIssue, itraceComp)) {
    t.io.clock       := clock
    t.io.reset       := reset.asBool
    t.io.is_issue    := 0.U
    t.io.rob_id      := 0.U
    t.io.domain_id   := 0.U
    t.io.funct       := 0.U
    t.io.pc          := 0.U
    t.io.rs1_idx     := 0.U
    t.io.rs2_idx     := 0.U
    t.io.rs1_data    := 0.U
    t.io.rs2_data    := 0.U
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

  val headPtr                = RegInit(0.U(idWidth.W))
  val tailPtr                = RegInit(0.U(idWidth.W))
  val issuedCount            = RegInit(0.U(log2Up(robDepth + 1).W))
  val bankCols               = RegInit(VecInit(Seq.fill(b.memDomain.virtualBankCount)(0.U(log2Up(b.memDomain.bankNum + 1).W))))
  // In-flight ownership is tracked in the architectural vbank namespace.
  private val vbankMaskWidth = b.memDomain.virtualBankCount
  val vbankBusy              = RegInit(0.U(vbankMaskWidth.W))

  // Dependency state is maintained incrementally instead of rebuilding all
  // older-entry comparisons every cycle. A bit in slot k names the ROB slot
  // that currently blocks k. Ordinary RAW dependencies disappear at complete;
  // dependencies involving a mapping/config instruction disappear at commit.
  // Keeping one union mask is sufficient because a config dependency is never
  // allowed to clear earlier than commit, even when it also looks like a RAW.
  val dependencyMask = RegInit(VecInit(Seq.fill(robDepth)(0.U(robDepth.W))))

  val isEmpty = headPtr === tailPtr && !robValid(headPtr)
  val isFull  = headPtr === tailPtr && robValid(headPtr)

  def nextPtr(p: UInt): UInt = Mux(p === (robDepth - 1).U, 0.U, p + 1.U)
  def wrapPtr(v: UInt): UInt = Mux(v >= robDepth.U, v - robDepth.U, v)
  def robIdx(v:  UInt): UInt = v(idWidth - 1, 0)

  def isMappingConfig(entry: GlobalRobEntry): Bool =
    entry.cmd.domain_id === DomainId.MEM &&
      (entry.cmd.cmd.funct === MSET_BITPAT)

  def accessUsesBank(access: BankAccessInfo, bank: UInt): Bool =
    (access.rd_bank_0_valid && access.rd_bank_0_id === bank) ||
      (access.rd_bank_1_valid && access.rd_bank_1_id === bank) ||
      (access.wr_bank_valid && access.wr_bank_id === bank)

  // Build one architectural-vbank mask per ROB slot once, then hazard checks
  // become mask intersections instead of repeating ID comparisons.
  private val hazardBankCount = b.memDomain.virtualBankCount

  def rawReadMask(access: BankAccessInfo): UInt = {
    val rd0 = Mux(
      access.rd_bank_0_valid && access.rd_bank_0_id < hazardBankCount.U,
      UIntToOH(access.rd_bank_0_id, hazardBankCount),
      0.U(hazardBankCount.W)
    )
    val rd1 = Mux(
      access.rd_bank_1_valid && access.rd_bank_1_id < hazardBankCount.U,
      UIntToOH(access.rd_bank_1_id, hazardBankCount),
      0.U(hazardBankCount.W)
    )
    rd0 | rd1
  }

  def rawWriteMask(access: BankAccessInfo): UInt = Mux(
    access.wr_bank_valid && access.wr_bank_id < hazardBankCount.U,
    UIntToOH(access.wr_bank_id, hazardBankCount),
    0.U(hazardBankCount.W)
  )

  def rawUseMask(access: BankAccessInfo): UInt =
    rawReadMask(access) | rawWriteMask(access)

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
  tailAlias :=
    b.memDomain.virtualBankCount.U(b.frontend.bank_id_len.W) + tailPtr

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

  // Bank masks are shared by allocation and issue. This avoids rebuilding the
  // UIntToOH decoders once for every older-entry comparison at allocation.
  val entryRawReads  = Wire(Vec(robDepth, UInt(hazardBankCount.W)))
  val entryRawWrites = Wire(Vec(robDepth, UInt(hazardBankCount.W)))
  val entryIsConfig  = Wire(Vec(robDepth, Bool()))
  for (slot <- 0 until robDepth) {
    entryRawReads(slot)  := rawReadMask(robEntries(slot).cmd.bankAccess)
    entryRawWrites(slot) := rawWriteMask(robEntries(slot).cmd.bankAccess)
    entryIsConfig(slot)  := isMappingConfig(robEntries(slot))
  }

  // A newly allocated entry is always younger than all currently valid ROB
  // entries (tailPtr points at the first free slot). Capture its older-entry
  // dependencies once here; issue no longer repeats pairwise comparisons.
  val allocReadMask       = rawReadMask(io.alloc.bits.bankAccess)
  val allocWriteMask      = rawWriteMask(io.alloc.bits.bankAccess)
  val allocIsConfig       = io.alloc.bits.domain_id === DomainId.MEM &&
    (io.alloc.bits.cmd.funct === MSET_BITPAT)
  val allocDependencyBits = Wire(Vec(robDepth, Bool()))
  for (older <- 0 until robDepth) {
    val olderUseMask   = entryRawReads(older) | entryRawWrites(older)
    val rawConflict    = ((allocReadMask & entryRawWrites(older)) |
      (allocWriteMask & olderUseMask)).orR
    val configConflict = (allocIsConfig || entryIsConfig(older)) &&
      ((allocReadMask | allocWriteMask) & olderUseMask).orR
    // Config conflicts include entries that have completed but are waiting to
    // commit. RAW conflicts only include entries that are still executing.
    allocDependencyBits(older) :=
      (robValid(older) && !robComplete(older) && rawConflict) ||
        (robValid(older) && configConflict)
  }
  val allocDependencies = allocDependencyBits.asUInt

  when(io.alloc.fire) {
    itraceAlloc.io.is_issue    := 2.U
    itraceAlloc.io.rob_id      := tailPtr
    itraceAlloc.io.domain_id   := io.alloc.bits.domain_id
    itraceAlloc.io.funct       := io.alloc.bits.cmd.funct
    itraceAlloc.io.pc          := io.alloc.bits.cmd.pc
    itraceAlloc.io.rs1_idx     := io.alloc.bits.cmd.rs1
    itraceAlloc.io.rs2_idx     := io.alloc.bits.cmd.rs2
    itraceAlloc.io.rs1_data    := io.alloc.bits.cmd.rs1Data
    itraceAlloc.io.rs2_data    := io.alloc.bits.cmd.rs2Data
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
    itraceComp.io.rs1_idx     := robEntries(cid).cmd.cmd.rs1
    itraceComp.io.rs2_idx     := robEntries(cid).cmd.cmd.rs2
    itraceComp.io.rs1_data    := robEntries(cid).cmd.cmd.rs1Data
    itraceComp.io.rs2_data    := robEntries(cid).cmd.cmd.rs2Data
    itraceComp.io.bank_enable := robEntries(cid).cmd.cmd.funct(6, 4)
    itraceComp.io.enable      := true.B
  }

  val completedBitMask = Mux(
    io.complete.fire,
    UIntToOH(io.complete.bits, robDepth),
    0.U(robDepth.W)
  )

  // ---------------------------------------------------------------------------
  // Issue: scan from head for first issuable entry (valid && !issued && !complete)
  // ---------------------------------------------------------------------------
  val scanValid  = Wire(Vec(robDepth, Bool()))
  val scanReady  = Wire(Vec(robDepth, Bool()))
  val vbankMasks = Wire(Vec(robDepth, UInt(vbankMaskWidth.W)))
  for (slot <- 0 until robDepth) {
    // Reuse the raw hazard mask for active-command ownership.
    vbankMasks(slot) := entryRawReads(slot) | entryRawWrites(slot)
  }

  // Check each candidate against older ROB entries using mask intersections.
  // The masks are precomputed once per slot, so this retains the original
  // age-sensitive RAW/WAR/WAW semantics without replicating three equality
  // comparators for every bank access pair.
  for (i <- 0 until robDepth) {
    val ptr = robIdx(wrapPtr(headPtr + i.U))
    scanValid(i)           := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
    scoreboard.queryVec(i) := robEntries(ptr).renamedBankAccess
    scanReady(i)           := scanValid(i) && !scoreboard.hazardVec(i) &&
      !dependencyMask(ptr).orR &&
      !(vbankMasks(ptr) & vbankBusy).orR
  }

  val hasReady       = scanReady.asUInt.orR
  val firstReady     = PriorityEncoder(scanReady.asUInt)
  val actualIssuePtr = robIdx(wrapPtr(headPtr + firstReady))

  scoreboard.query := robEntries(actualIssuePtr).renamedBankAccess
  val canIssue = hasReady

  val issueStageValid = RegInit(false.B)
  val issueStageEntry = Reg(new GlobalRobEntry(b))
  val issueStageFire  = issueStageValid && !io.subRobActive && io.issue.ready
  val issueStageReady = !issueStageValid || issueStageFire

  // Keep the wide ROB entry and bank-column mux one register boundary away
  // from the age/scoreboard scan. The pointer stage can accept one candidate
  // per cycle; the payload stage drains at one per cycle, so steady-state II
  // remains one while the extra latency is hidden by in-flight commands.
  val issuePtrValid = RegInit(false.B)
  val issuePtr      = Reg(UInt(idWidth.W))
  val issuePtrFire  = issuePtrValid && issueStageReady
  val issuePtrReady = !issuePtrValid || issuePtrFire
  val issueLoad     = canIssue && !io.subRobActive && issuePtrReady
  val issuePayload  = Wire(new GlobalRobEntry(b))
  issuePayload             := robEntries(issuePtr)
  issuePayload.cmd.op1_col := Mux(
    issuePayload.cmd.bankAccess.rd_bank_0_valid,
    bankCols(issuePayload.cmd.bankAccess.rd_bank_0_id(b.memDomain.vbankIdWidth - 1, 0)),
    0.U
  )
  issuePayload.cmd.op2_col := Mux(
    issuePayload.cmd.bankAccess.rd_bank_1_valid,
    bankCols(issuePayload.cmd.bankAccess.rd_bank_1_id(b.memDomain.vbankIdWidth - 1, 0)),
    0.U
  )
  issuePayload.cmd.wr_col  := Mux(
    issuePayload.cmd.bankAccess.wr_bank_valid,
    bankCols(issuePayload.cmd.bankAccess.wr_bank_id(b.memDomain.vbankIdWidth - 1, 0)),
    0.U
  )

  io.issue.valid := issueStageValid && !io.subRobActive
  io.issue.bits  := issueStageEntry

  when(issuePtrReady) {
    issuePtrValid := issueLoad
    when(issueLoad) {
      issuePtr := actualIssuePtr
    }
  }
  when(issueStageReady) {
    issueStageValid := issuePtrValid
    when(issuePtrFire) {
      issueStageEntry := issuePayload
    }
  }

  scoreboard.issue.valid := false.B
  scoreboard.issue.bits  := 0.U.asTypeOf(scoreboard.issue.bits)

  when(issueLoad) {
    robIssued(actualIssuePtr) := true.B
    issueFired                := true.B
    scoreboard.issue.valid    := true.B
    scoreboard.issue.bits     := robEntries(actualIssuePtr).renamedBankAccess

    itraceIssue.io.is_issue    := 1.U
    itraceIssue.io.rob_id      := robEntries(actualIssuePtr).rob_id
    itraceIssue.io.domain_id   := robEntries(actualIssuePtr).cmd.domain_id
    itraceIssue.io.funct       := robEntries(actualIssuePtr).cmd.cmd.funct
    itraceIssue.io.pc          := robEntries(actualIssuePtr).cmd.cmd.pc
    itraceIssue.io.rs1_idx     := robEntries(actualIssuePtr).cmd.cmd.rs1
    itraceIssue.io.rs2_idx     := robEntries(actualIssuePtr).cmd.cmd.rs2
    itraceIssue.io.rs1_data    := robEntries(actualIssuePtr).cmd.cmd.rs1Data
    itraceIssue.io.rs2_data    := robEntries(actualIssuePtr).cmd.cmd.rs2Data
    itraceIssue.io.bank_enable := robEntries(actualIssuePtr).cmd.cmd.funct(6, 4)
    itraceIssue.io.enable      := true.B
  }

  // Preserve the original update ordering when issue and complete coincide:
  // a completion clears ownership after the newly issued transaction marks
  // its banks busy.
  val issuedVbankMask    = Mux(issueLoad, vbankMasks(actualIssuePtr), 0.U)
  val completedVbankMask = Mux(io.complete.fire, vbankMasks(io.complete.bits), 0.U)
  when(issueLoad || io.complete.fire) {
    vbankBusy :=
      (vbankBusy & ~completedVbankMask) | issuedVbankMask
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
        val bank = robEntries(i).cmd.bankAccess.wr_bank_id(b.memDomain.vbankIdWidth - 1, 0)
        val col  = robEntries(i).cmd.cmd.rs2Data(9, 5)
        when(robEntries(i).cmd.cmd.rs2Data(10)) {
          bankCols(bank) := Mux(col === 0.U, b.memDomain.bankNum.U, col)
        }.otherwise {
          bankCols(bank) := 0.U
        }
      }
      robValid(i)    := false.B
      robIssued(i)   := false.B
      robComplete(i) := false.B
    }
  }

  // Maintain dependency masks at the same architectural events that gate the
  // issue scan. A dependency involving a config entry (on either side) clears
  // only at commit; all other dependencies clear at complete. This is the
  // union-mask form of the former separate RAW/config masks.
  val configEntryMask = entryIsConfig.asUInt
  for (slot <- 0 until robDepth) {
    val isAllocatedSlot = io.alloc.fire && tailPtr === slot.U
    val slotIsConfig    = Mux(isAllocatedSlot, allocIsConfig, entryIsConfig(slot))
    // Each dependency bit names the older entry that blocks this slot. Keep a
    // config-related edge through completion when either endpoint is config.
    val configRelated   = configEntryMask | Fill(robDepth, slotIsConfig)
    val clearMask       = commitMask.asUInt | (completedBitMask & ~configRelated)
    when(isAllocatedSlot) {
      dependencyMask(slot) := allocDependencies & ~clearMask
    }.otherwise {
      dependencyMask(slot) := dependencyMask(slot) & ~clearMask
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
