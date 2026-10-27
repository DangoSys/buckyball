package framework.acc.frontend.scheduler.rob.global

import framework.lib.arbiter.{Arbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.decoder.{DecodedInstruction, DomainId, PostGDCmd}
import framework.acc.frontend.scheduler.scoreboard.BankAccessInfo
import framework.acc.memdomain.DISA.MSET_BITPAT
import framework.acc.memdomain.DISA.MVIN_KERNEL_BITPAT
import framework.acc.gpdomain.decoder.DISA.RUN_KERNEL_BITPAT
import framework.acc.memdomain.bank_router.{BankMapping, BankMappingIO}
import framework.acc.memdomain.{MemoryContext, PhysicalBankHash}
import framework.acc.debug.trace.InstructionEvent
import framework.acc.debug.DebugPort

class GlobalRobEntry(val frontendParam: FrontendParam, val memParam: MemDomainParam, val permissionParam: PmpParams)
    extends Bundle {
  val commandDataBits   = frontendParam.commandDataBits
  val context           = new MemoryContext(permissionParam)
  val cmd               = new PostGDCmd(frontendParam, memParam)
  val renamedBankAccess = new BankAccessInfo(frontendParam.bank_id_len)
  val rob_id            = UInt(log2Up(frontendParam.rob_entries).W)
  val instId            = UInt(64.W)
}

@instantiable
class GlobalROB(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val permissionParam: PmpParams,
  val diffTest:        Boolean)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val ownerId         = frontendParam.ownerId

  require(
    !memParam.sharedEnable || (frontendParam.bank_alias_enable && memParam.bankLayoutEnable),
    "BAT/BLT ablation modes require private-bank workloads"
  )

  val robDepth = frontendParam.rob_entries
  val idWidth  = log2Up(robDepth)

  val resourceCount =
    memParam.virtualBankCount + (if (gpParam.enable) 2
                                 else 0)

  val scoreBankNum      = 2 * resourceCount + robDepth
  require(
    scoreBankNum <= (1 << frontendParam.bank_id_len),
    "Bank resource IDs and ROB aliases exceed bank_id_len"
  )
  val sharedHashCount   =
    if (memParam.sharedEnable) memParam.sharedBankNum else 0
  val physicalBankCount = memParam.bankNum + sharedHashCount

  require(
    frontendParam.max_private_vbank < memParam.virtualBankCount,
    s"max_private_vbank(${frontendParam.max_private_vbank}) must be < virtualBankCount(${memParam.virtualBankCount})"
  )

  @public
  val io = IO(new Bundle {
    val traceInstructions = Output(Vec(4, new InstructionEvent))
    val alloc             =
      Flipped(new DecoupledIO(new DecodedInstruction(frontendParam, memParam, permissionParam)))
    val issue             = new DecoupledIO(new GlobalRobEntry(frontendParam, memParam, permissionParam))
    val complete          = Flipped(new DecoupledIO(UInt(idWidth.W)))
    val mapping           = new BankMappingIO(frontendParam)

    val bat = new Bundle {

      val alloc = Output(new Bundle {
        val valid       = Bool()
        val rename      = Bool()
        val reset_read0 = Bool()
        val raw         = new BankAccessInfo(frontendParam.bank_id_len)
      })

      val reclaim = Output(Valid(UInt(frontendParam.bank_id_len.W)))
      val stale   = Input(Vec(2, Valid(UInt(frontendParam.bank_id_len.W))))
      val ready   = Input(Bool())
      val renamed = Input(new BankAccessInfo(frontendParam.bank_id_len))
    }

    val scoreboard = new Bundle {
      val alloc    = Valid(new BankAccessInfo(frontendParam.bank_id_len))
      val issue    = Valid(new BankAccessInfo(frontendParam.bank_id_len))
      val complete = Valid(new BankAccessInfo(frontendParam.bank_id_len))

      val query = Output(Vec(
        robDepth,
        new BankAccessInfo(frontendParam.bank_id_len)
      ))

      val hazard = Input(Vec(robDepth, Bool()))
    }

    val empty = Output(Bool())
    val full  = Output(Bool())

    val debug = Option.when(diffTest)(new DebugPort(frontendParam))

  })

  val mappingIssue  = Wire(Decoupled(new BankMapping(frontendParam)))
  val mappingRetire = Wire(Decoupled(new BankMapping(frontendParam)))
  val mappings      = Instantiate(new Arbiter(new BankMapping(frontendParam), 2))
  mappings.io.in(0) <> mappingRetire
  mappings.io.in(1) <> mappingIssue
  io.mapping.valid      := mappings.io.out.valid
  io.mapping.bits       := mappings.io.out.bits
  mappings.io.out.ready := io.mapping.ready

  // ---------------------------------------------------------------------------
  // Passive instruction observations consumed by the thread trace collector.
  // ---------------------------------------------------------------------------
  val itraceAlloc = io.traceInstructions(0)
  val itraceIssue = io.traceInstructions(1)
  val itraceComp  = io.traceInstructions(2)

  for (t <- Seq(itraceAlloc, itraceIssue, itraceComp)) {
    t.is_issue  := 0.U
    t.thread_id := ownerId.U(64.W)
    t.rob_id    := 0.U
    t.inst_id   := 0.U
    t.domain_id := 0.U
    t.funct     := 0.U
    t.pc        := 0.U
    t.rs1_data  := 0.U
    t.rs2_data  := 0.U
    t.enable    := false.B
  }

  // ---------------------------------------------------------------------------
  // Storage
  // ---------------------------------------------------------------------------
  val queue: Instance[Rob] = Instantiate(new Rob(configs.RobParams(
    robDepth,
    (new GlobalRobEntry(frontendParam, memParam, permissionParam)).getWidth,
    if (diffTest) 1 else robDepth
  )))

  val robEntries    =
    VecInit(queue.io.data.map(_.asTypeOf(new GlobalRobEntry(frontendParam, memParam, permissionParam))))
  val robValid      = queue.io.valid
  val robIssued     = queue.io.issued
  val robComplete   = queue.io.completed
  val allocatedBank = RegInit(VecInit(Seq.fill(robDepth)(false.B)))
  val nextInstId    = RegInit(1.U(64.W))

  val headPtr     = queue.io.head
  val tailPtr     = queue.io.tail
  val issuedCount = queue.io.issuedCount

  val dependencies: Instance[DependencyResolver] =
    Instantiate(new DependencyResolver(frontendParam, memParam, gpParam, permissionParam))
  dependencies.io.alloc    := io.alloc.bits.operation
  dependencies.io.entries  := robEntries
  dependencies.io.valid    := robValid
  dependencies.io.complete := robComplete

  // Dependency state is maintained incrementally instead of rebuilding all
  // older-entry comparisons every cycle. A bit in slot k names the ROB slot
  // that currently blocks k. Ordinary RAW dependencies disappear at complete;
  // dependencies involving a mapping/config instruction disappear at commit.
  // Keeping one union mask is sufficient because a config dependency is never
  // allowed to clear earlier than commit, even when it also looks like a RAW.
  val dependencyMask = queue.io.dependencies

  val isEmpty                    = queue.io.empty
  val isFull                     = queue.io.full
  val fullRequestBlocked         = isFull && io.alloc.valid
  val previousFullRequestBlocked = RegNext(fullRequestBlocked, false.B)
  val itraceRobFull              = io.traceInstructions(3)
  itraceRobFull.is_issue  := Mux(fullRequestBlocked, 3.U, 4.U)
  itraceRobFull.thread_id := ownerId.U(64.W)
  itraceRobFull.rob_id    := tailPtr
  itraceRobFull.inst_id   := 0.U
  itraceRobFull.domain_id := 0.U
  itraceRobFull.funct     := 0.U
  itraceRobFull.pc        := io.alloc.bits.operation.cmd.pc
  itraceRobFull.rs1_data  := 0.U
  itraceRobFull.rs2_data  := 0.U
  itraceRobFull.enable    := fullRequestBlocked =/= previousFullRequestBlocked

  // ---------------------------------------------------------------------------
  // Allocate: enqueue decoded instruction into ROB
  // rob_id == tailPtr at allocation time (no separate counter needed)
  // ---------------------------------------------------------------------------
  val commitMask = queue.io.retired.asBools
  val staleValid = RegInit(VecInit(Seq.fill(robDepth)(0.U(2.W))))
  val staleBank  = Reg(Vec(robDepth, Vec(2, UInt(frontendParam.bank_id_len.W))))
  val staleRaw   = Reg(Vec(robDepth, UInt(frontendParam.bank_id_len.W)))
  val newBank    = Reg(Vec(robDepth, UInt(frontendParam.bank_id_len.W)))

  val retireSlots = VecInit((0 until robDepth).map { offset =>
    val sum = headPtr +& offset.U
    Mux(sum >= robDepth.U, sum - robDepth.U, sum)(idWidth - 1, 0)
  })

  val pendingReclaim = VecInit(retireSlots.map(slot => queue.io.retireCandidates(slot) && staleValid(slot).orR))
  val reclaimSlot    = retireSlots(PriorityEncoder(pendingReclaim))
  val reclaimIndex   = PriorityEncoder(staleValid(reclaimSlot))
  val targetRaw      = robEntries(reclaimSlot).cmd.bankAccess.wr_bank_id
  mappingRetire.valid         := pendingReclaim.asUInt.orR
  mappingRetire.bits.bank     := Mux(reclaimIndex === 0.U, staleRaw(reclaimSlot), targetRaw)
  mappingRetire.bits.vbank    := Mux(reclaimIndex === 0.U, newBank(reclaimSlot), targetRaw)
  mappingRetire.bits.previous := staleBank(reclaimSlot)(reclaimIndex)
  mappingRetire.bits.threadId := ownerId.U(64.W)
  mappingRetire.bits.allocate := false.B
  mappingRetire.bits.reuse    := false.B
  mappingRetire.bits.retire   := true.B
  mappingRetire.bits.write    := false.B
  when(mappingRetire.fire) {
    staleValid(reclaimSlot) := staleValid(reclaimSlot) & ~UIntToOH(reclaimIndex, 2)
  }
  for (slot <- 0 until robDepth) {
    queue.io.retireReady(slot)              := !staleValid(slot).orR ||
      (slot.U === reclaimSlot && mappingRetire.ready && PopCount(staleValid(slot)) === 1.U)
    when(commitMask(slot))(staleValid(slot) := 0.U)
  }
  io.bat.reclaim.valid        := mappingRetire.fire
  io.bat.reclaim.bits         := mappingRetire.bits.previous

  val headNeedsDebug =
    robEntries(headPtr).instId =/= 0.U && robEntries(headPtr).cmd.cmd.raw_inst(31, 25) =/= MSET_BITPAT &&
      robEntries(headPtr).cmd.bankAccess.wr_bank_valid && robEntries(
        headPtr
      ).cmd.bankAccess.wr_bank_id < memParam.virtualBankCount.U

  val hasCommit = queue.io.retired.orR

  val admit = io.bat.ready && !dependencies.io.movementLive
  io.alloc.ready           := queue.io.alloc.ready && admit
  queue.io.alloc.valid     := io.alloc.valid && admit
  io.bat.alloc.valid       := io.alloc.fire
  io.bat.alloc.raw         := io.alloc.bits.operation.bankAccess
  io.bat.alloc.rename      := frontendParam.bank_alias_enable.B && io.alloc.bits.operation.bankAccess.wr_bank_valid &&
    io.alloc.bits.operation.bankAccess.wr_bank_id < memParam.virtualBankCount.U &&
    io.alloc.bits.operation.cmd.raw_inst(31, 25) =/= MSET_BITPAT && !io.alloc.bits.operation.bankAccess.wr_update
  io.bat.alloc.reset_read0 := io.alloc.bits.operation.cmd.raw_inst(
    31,
    25
  ) === MSET_BITPAT && io.alloc.bits.operation.cmd
    .rs2Data(12)

  // Mark write alias as busy in scoreboard at alloc time (not issue time).
  io.scoreboard.alloc.valid := io.alloc.fire && io.bat.alloc.rename
  io.scoreboard.alloc.bits  := io.bat.renamed

  val allocIsConfig   = io.alloc.bits.operation.cmd.raw_inst(31, 25) === MSET_BITPAT
  val allocUsesInstId = io.alloc.bits.operation.runtime
  val allocation      = Wire(new GlobalRobEntry(frontendParam, memParam, permissionParam))
  allocation.context                := io.alloc.bits.context
  allocation.cmd                    := io.alloc.bits.operation
  allocation.renamedBankAccess      := io.bat.renamed
  allocation.rob_id                 := tailPtr
  allocation.instId                 := Mux(allocUsesInstId, nextInstId, 0.U)
  queue.io.alloc.bits.data          := allocation.asUInt
  queue.io.alloc.bits.dependencies  := dependencies.io.dependencies
  queue.io.alloc.bits.commitBarrier := allocIsConfig

  when(io.alloc.fire) {
    itraceAlloc.is_issue  := 2.U
    itraceAlloc.rob_id    := tailPtr
    itraceAlloc.inst_id   := allocation.instId
    itraceAlloc.domain_id := io.alloc.bits.operation.domain_id
    itraceAlloc.funct     := io.alloc.bits.operation.cmd.raw_inst(31, 25)
    itraceAlloc.pc        := io.alloc.bits.operation.cmd.pc
    itraceAlloc.rs1_data  := io.alloc.bits.operation.cmd.rs1Data
    itraceAlloc.rs2_data  := io.alloc.bits.operation.cmd.rs2Data
    itraceAlloc.enable    := true.B

    allocatedBank(tailPtr) := io.bat.alloc.rename
    staleValid(tailPtr)    := VecInit(io.bat.stale.map(_.valid)).asUInt
    staleBank(tailPtr)     := VecInit(io.bat.stale.map(_.bits))
    staleRaw(tailPtr)      := Mux(
      io.bat.alloc.reset_read0,
      io.alloc.bits.operation.bankAccess.rd_bank_0_id,
      io.alloc.bits.operation.bankAccess.wr_bank_id
    )
    newBank(tailPtr)       := Mux(
      io.bat.alloc.reset_read0,
      io.alloc.bits.operation.bankAccess.rd_bank_0_id,
      io.bat.renamed.wr_bank_id
    )

    when(allocUsesInstId) {
      nextInstId := nextInstId + 1.U
    }
  }

  // ---------------------------------------------------------------------------
  // Complete: mark entry as completed, release scoreboard resources
  // ---------------------------------------------------------------------------
  if (diffTest) {
    val debug          = io.debug.get
    val incomingAccess = robEntries(io.complete.bits).cmd.bankAccess
    val needsCheck     = robEntries(io.complete.bits).instId =/= 0.U &&
      !(robEntries(io.complete.bits).cmd.domain_id === DomainId.MEM &&
        robEntries(io.complete.bits).cmd.cmd.raw_inst(31, 25) === MSET_BITPAT) &&
      incomingAccess.wr_bank_valid && incomingAccess.wr_bank_id < memParam.virtualBankCount.U
    debug.check.valid      := io.complete.valid && needsCheck
    debug.check.bits.robId := io.complete.bits
    debug.check.bits.bank  := (if (memParam.bankLayoutEnable)
                                robEntries(io.complete.bits).renamedBankAccess.wr_bank_id
                              else incomingAccess.wr_bank_id)
    io.complete.ready      := !needsCheck || (debug.checked.valid && debug.checked.bits === io.complete.bits)
    debug.checked.ready    := io.complete.fire && needsCheck
  } else {
    io.complete.ready := true.B
  }

  io.scoreboard.complete.valid := false.B
  io.scoreboard.complete.bits  := 0.U.asTypeOf(io.scoreboard.complete.bits)

  val issueFired = WireDefault(false.B)
  queue.io.complete.valid := io.complete.fire
  queue.io.complete.bits  := io.complete.bits

  when(io.complete.fire) {
    val cid = io.complete.bits
    io.scoreboard.complete.valid := true.B
    io.scoreboard.complete.bits  := robEntries(cid).renamedBankAccess

    itraceComp.is_issue  := 0.U
    itraceComp.rob_id    := cid
    itraceComp.inst_id   := robEntries(cid).instId
    itraceComp.domain_id := robEntries(cid).cmd.domain_id
    itraceComp.funct     := robEntries(cid).cmd.cmd.raw_inst(31, 25)
    itraceComp.pc        := robEntries(cid).cmd.cmd.pc
    itraceComp.rs1_data  := robEntries(cid).cmd.cmd.rs1Data
    itraceComp.rs2_data  := robEntries(cid).cmd.cmd.rs2Data
    itraceComp.enable    := true.B
  }

  // ---------------------------------------------------------------------------
  // Issue: select the oldest unissued entry; a blocked entry stalls younger entries.
  // ---------------------------------------------------------------------------
  val scanValid = Wire(Vec(robDepth, Bool()))
  val scanReady = Wire(Vec(robDepth, Bool()))
  // Check each candidate against older ROB entries using mask intersections.
  // The masks are precomputed once per slot, so this retains the original
  // age-sensitive RAW/WAR/WAW semantics without replicating three equality
  // comparators for every bank access pair.
  for (i <- 0 until robDepth) {
    val age = headPtr +& i.U
    val ptr = Mux(age >= robDepth.U, age - robDepth.U, age)(idWidth - 1, 0)
    scanValid(i)           := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
    io.scoreboard.query(i) := robEntries(ptr).renamedBankAccess
    scanReady(i)           := scanValid(i) && !io.scoreboard.hazard(i) && !dependencyMask(
      ptr
    ).orR
  }

  val firstReady     = PriorityEncoder(scanValid.asUInt)
  val hasReady       = scanValid.asUInt.orR && scanReady(firstReady)
  val issueAge       = headPtr +& firstReady
  val actualIssuePtr = Mux(issueAge >= robDepth.U, issueAge - robDepth.U, issueAge)(idWidth - 1, 0)

  val canIssue = hasReady

  val issueStageValid = RegInit(false.B)
  val issueStageEntry = Reg(new GlobalRobEntry(frontendParam, memParam, permissionParam))
  val issueStageFire  = issueStageValid && io.issue.ready
  val issueStageReady = !issueStageValid || issueStageFire

  // Keep the wide ROB entry and bank-column mux one register boundary away
  // from the age/scoreboard scan. The pointer stage can accept one candidate
  // per cycle; the payload stage drains at one per cycle, so steady-state II
  // remains one while the extra latency is hidden by in-flight commands.
  val issuePtrValid = RegInit(false.B)
  val issuePtr      = Reg(UInt(idWidth.W))
  // The BLT is still installing the mapping this instruction issued; hold it.
  val issuePtrFire  = issuePtrValid && issueStageReady && !io.mapping.busy
  val issuePtrReady = !issuePtrValid || issuePtrFire
  mappingIssue.bits.bank  := robEntries(actualIssuePtr).cmd.bankAccess.wr_bank_id
  mappingIssue.bits.vbank := robEntries(actualIssuePtr).renamedBankAccess.wr_bank_id
  val issueAccess   = robEntries(actualIssuePtr).renamedBankAccess
  val previousAlias = staleBank(actualIssuePtr)(0)

  val readsPrevious = (issueAccess.rd_bank_0_valid && issueAccess.rd_bank_0_id === previousAlias) ||
    (issueAccess.rd_bank_1_valid && issueAccess.rd_bank_1_id === previousAlias) ||
    (issueAccess.rd_wr_valid && issueAccess.rd_wr_id === previousAlias)

  // At the ROB head all older users have retired; younger users see the new alias.
  mappingIssue.bits.previous := previousAlias
  mappingIssue.bits.reuse    := allocatedBank(actualIssuePtr) && actualIssuePtr === headPtr && !readsPrevious
  mappingIssue.bits.allocate := allocatedBank(actualIssuePtr)
  mappingIssue.bits.retire   := false.B
  mappingIssue.bits.write    := robEntries(actualIssuePtr).cmd.bankAccess.wr_bank_valid &&
    !robEntries(actualIssuePtr).cmd.bankAccess.wr_update &&
    robEntries(
      actualIssuePtr
    ).cmd.bankAccess.wr_bank_id < memParam.virtualBankCount.U &&
    robEntries(actualIssuePtr).cmd.cmd.raw_inst(31, 25) =/= MSET_BITPAT
  mappingIssue.bits.threadId := ownerId.U(64.W)
  val needsMapping = mappingIssue.bits.allocate || mappingIssue.bits.write
  mappingIssue.valid := canIssue && issuePtrReady && needsMapping
  val issueLoad    = canIssue && issuePtrReady && (!needsMapping || mappingIssue.ready)
  val issuePayload = Wire(new GlobalRobEntry(frontendParam, memParam, permissionParam))
  issuePayload := robEntries(issuePtr)

  io.issue.valid := issueStageValid
  io.issue.bits  := issueStageEntry

  when(issuePtrReady) {
    issuePtrValid := issueLoad
    when(issueLoad) {
      issuePtr := actualIssuePtr
    }
  }
  when(issueStageReady) {
    issueStageValid := issuePtrFire
    when(issuePtrFire) {
      issueStageEntry := issuePayload
    }
  }

  io.scoreboard.issue.valid := false.B
  io.scoreboard.issue.bits  := 0.U.asTypeOf(io.scoreboard.issue.bits)

  when(issueLoad) {
    issueFired                := true.B
    io.scoreboard.issue.valid := true.B
    io.scoreboard.issue.bits  := robEntries(actualIssuePtr).renamedBankAccess

    itraceIssue.is_issue  := 1.U
    itraceIssue.rob_id    := robEntries(actualIssuePtr).rob_id
    itraceIssue.inst_id   := robEntries(actualIssuePtr).instId
    itraceIssue.domain_id := robEntries(actualIssuePtr).cmd.domain_id
    itraceIssue.funct     := robEntries(actualIssuePtr).cmd.cmd.raw_inst(31, 25)
    itraceIssue.pc        := robEntries(actualIssuePtr).cmd.cmd.pc
    itraceIssue.rs1_data  := robEntries(actualIssuePtr).cmd.cmd.rs1Data
    itraceIssue.rs2_data  := robEntries(actualIssuePtr).cmd.cmd.rs2Data
    itraceIssue.enable    := true.B
  }

  queue.io.issue.valid := issueFired
  queue.io.issue.bits  := actualIssuePtr

  io.debug.foreach { debug =>
    debug.commit.valid         := hasCommit && headNeedsDebug
    debug.commit.bits.robId    := headPtr
    debug.commit.bits.instId   := robEntries(headPtr).instId
    debug.commit.bits.threadId := ownerId.U(64.W)
    debug.commit.bits.bank     := robEntries(headPtr).cmd.bankAccess.wr_bank_id
    debug.idle                 := io.empty && !io.alloc.valid && !debug.commit.valid
  }

  val mappingCommits = VecInit(
    (0 until robDepth).map(i =>
      commitMask(i) && robEntries(i).cmd.domain_id === DomainId.MEM &&
        robEntries(i).cmd.cmd.raw_inst(31, 25) === MSET_BITPAT
    )
  )

  assert(
    PopCount(mappingCommits) <= 1.U,
    "Mapping dependency permits one configuration commit"
  )
  // ---------------------------------------------------------------------------
  // Status outputs
  // ---------------------------------------------------------------------------
  io.empty := isEmpty
  io.full  := isFull

}
