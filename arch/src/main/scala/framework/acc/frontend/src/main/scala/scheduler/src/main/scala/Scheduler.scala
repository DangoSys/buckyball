package framework.acc.frontend.scheduler

import sims.configs.SimParam
import framework.lib.arbiter.{Arbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import framework.lib.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.frontend.scheduler.rob.global.GlobalROB
import framework.acc.frontend.decoder.{DecodedInstruction, DomainId, PostGDCmd}
import framework.acc.frontend.scheduler.bat.BankAliasTable
import framework.acc.frontend.scheduler.scoreboard.{BankAccessInfo, BankScoreboard}
import framework.acc.debug.trace.InstructionEvent
import framework.acc.debug.DebugPort
import framework.acc.memdomain.bank_router.BankMapping
import framework.acc.memdomain.{Response => MemoryResponse, MemoryContext}
import framework.acc.memdomain.toddr.DmaError

/** Work dispatched to a domain, independent of the ROB's storage representation. */
class Issue(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val commandDataBits   = frontendParam.commandDataBits
  val p                 = frontendParam
  val cmd               = new PostGDCmd(frontendParam, memParam)
  val renamedBankAccess = new BankAccessInfo(frontendParam.bank_id_len)
  val rob_id            = UInt(log2Up(p.rob_entries).W)
  val instId            = UInt(64.W) // Observation identity; never controls scheduling.
  val target            = UInt(5.W)
}

/** Memory dispatch alone carries the submitting address-space state. */
class MemoryIssue(frontendParam: FrontendParam, memParam: MemDomainParam, permissionParam: PmpParams) extends Bundle {
  val commandDataBits = frontendParam.commandDataBits
  val operation       = new Issue(frontendParam, memParam)
  val context         = new MemoryContext(permissionParam)
}

/** Completion of an admitted operation; legality is an issue-time contract. */
class Complete(frontendParam: FrontendParam) extends Bundle {
  val rob_id = UInt(log2Up(frontendParam.rob_entries).W)
}

@instantiable
class Scheduler(
  val frontendParam:   FrontendParam,
  val memParam:        MemDomainParam,
  val gpParam:         GpDomainParam,
  val permissionParam: PmpParams,
  val simp:            SimParam)
    extends Module {
  val diffTest        = simp.diffTest
  val commandDataBits = frontendParam.commandDataBits
  val ownerId         = frontendParam.ownerId

  @public
  val io = IO(new Bundle {

    val admission = new Bundle {
      val command = Flipped(Decoupled(new DecodedInstruction(frontendParam, memParam, permissionParam)))
      val idle    = Output(Bool())
    }

    val ballRequest      = Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam))
    val ballCompletion   = Flipped(Decoupled(new MemoryResponse(frontendParam)))
    val memoryRequest    = Decoupled(new MemoryIssue(frontendParam, memParam, permissionParam))
    val memoryCompletion = Flipped(Decoupled(new MemoryResponse(frontendParam)))
    val gpRequest        = Option.when(gpParam.enable)(Decoupled(new Issue(frontendParam, memParam)))
    val gpCompletion     =
      Option.when(gpParam.enable)(Flipped(Decoupled(new Complete(frontendParam))))
    val mappingUpdate    = Decoupled(new BankMapping(frontendParam))

    val barrier = new Bundle {
      val arrive  = Output(Bool())
      val release = Input(Bool())
    }

    val fault             = Output(Bool())
    val traceInstructions = Output(Vec(4, new InstructionEvent))
    val debug             = Option.when(diffTest)(new DebugPort(frontendParam))
  })

  val rob: Instance[GlobalROB] =
    Instantiate(new GlobalROB(frontendParam, memParam, gpParam, permissionParam, simp.diffTest))

  val resourceCount = memParam.virtualBankCount + (if (gpParam.enable) 2 else 0)

  val bat: Instance[BankAliasTable] = Instantiate(
    new BankAliasTable(framework.acc.frontend.scheduler.bat.configs.Params(
      frontendParam.bank_id_len,
      resourceCount,
      frontendParam.rob_entries
    ))
  )

  val scoreboard: Instance[BankScoreboard] = Instantiate(
    new BankScoreboard(framework.acc.frontend.scheduler.scoreboard.configs.Params(
      frontendParam.bank_id_len,
      2 * resourceCount + frontendParam.rob_entries,
      frontendParam.rob_entries
    ))
  )

  scoreboard.io.alloc      := rob.io.scoreboard.alloc
  scoreboard.io.issue      := rob.io.scoreboard.issue
  scoreboard.io.complete   := rob.io.scoreboard.complete
  scoreboard.io.queryVec   := rob.io.scoreboard.query
  rob.io.scoreboard.hazard := scoreboard.io.hazardVec
  bat.io.alloc             := rob.io.bat.alloc
  bat.io.reclaim           := rob.io.bat.reclaim
  rob.io.bat.stale         := bat.io.stale
  rob.io.bat.ready         := bat.io.ready
  rob.io.bat.renamed       := bat.io.alloc_renamed
  io.mappingUpdate <> rob.io.mapping

  io.traceInstructions := rob.io.traceInstructions
  io.debug.zip(rob.io.debug).foreach { case (port, source) =>
    port <> source
    port.idle := io.admission.idle && !io.admission.command.valid && !port.commit.valid
  }

  val failed       = RegInit(false.B)
  val memoryFailed = io.memoryCompletion.bits.fault.error =/= DmaError.None.U
  when(
    (io.memoryCompletion.fire && memoryFailed) || (io.ballCompletion.fire && io.ballCompletion.bits.fault.error =/= 0.U)
  )(failed := true.B)
  io.fault := failed

  val isBarrierCmd       = io.admission.command.valid && io.admission.command.bits.operation.isBarrier
  val barrierWaitROB     = RegInit(false.B)
  val barrierWaitRelease = RegInit(false.B)
  when(isBarrierCmd && !barrierWaitROB && !barrierWaitRelease) {
    barrierWaitROB := true.B
  }
  when(barrierWaitROB && rob.io.empty) {
    barrierWaitROB     := false.B
    barrierWaitRelease := true.B
  }
  when(barrierWaitRelease && io.barrier.release) {
    barrierWaitRelease := false.B
  }
  io.barrier.arrive := barrierWaitRelease

  val isFrontendCmd   = io.admission.command.bits.operation.isBarrier
  // Extended control operations and metadata loads own the local scratch state until retirement.
  val funct           = io.admission.command.bits.operation.cmd.raw_inst(31, 25)
  val exclusive       = funct >= 80.U || funct === 35.U
  val exclusiveActive = RegInit(false.B)
  when(exclusiveActive && rob.io.empty)(exclusiveActive := false.B)
  when(rob.io.alloc.fire && exclusive)(exclusiveActive  := true.B)
  val anyStall = barrierWaitROB || barrierWaitRelease || failed || exclusiveActive || (exclusive && !rob.io.empty)
  rob.io.alloc.valid         := io.admission.command.valid && !isFrontendCmd && !anyStall
  rob.io.alloc.bits          := io.admission.command.bits
  io.admission.command.ready := Mux(
    isFrontendCmd,
    !anyStall,
    rob.io.alloc.ready && !anyStall
  )

  val is_ball_domain   = rob.io.issue.bits.cmd.domain_id === DomainId.BALL
  val is_mem_domain    = rob.io.issue.bits.cmd.domain_id === DomainId.MEM
  val is_kernel_domain = rob.io.issue.bits.cmd.domain_id === DomainId.GP

  val mainIssueEntry = Wire(new Issue(frontendParam, memParam))
  mainIssueEntry.instId            := rob.io.issue.bits.instId
  mainIssueEntry.target            := 0.U
  mainIssueEntry.cmd               := rob.io.issue.bits.cmd
  mainIssueEntry.renamedBankAccess := rob.io.issue.bits.renamedBankAccess
  mainIssueEntry.rob_id            := rob.io.issue.bits.rob_id
  mainIssueEntry.cmd.wr_read_bank  := rob.io.issue.bits.renamedBankAccess.rd_wr_id
  val rawOperands = rob.io.issue.bits.cmd.cmd.rs1Data
  val access      = rob.io.issue.bits.renamedBankAccess
  when(
    rob.io.issue.bits.cmd.cmd.raw_inst(31, 25) =/= framework.acc.memdomain.DISA.MSET_BITPAT
  ) {
    mainIssueEntry.cmd.cmd.rs1Data := Cat(
      rawOperands(63, 30),
      Mux(access.wr_bank_valid, access.wr_bank_id, rawOperands(29, 20)),
      Mux(access.rd_bank_1_valid, access.rd_bank_1_id, rawOperands(19, 10)),
      Mux(access.rd_bank_0_valid, access.rd_bank_0_id, rawOperands(9, 0))
    )
  }

  if (gpParam.enable) {
    io.gpRequest.get.valid := rob.io.issue.valid && is_kernel_domain && !failed
    io.gpRequest.get.bits  := mainIssueEntry
  }
  val kernelReady = if (gpParam.enable) io.gpRequest.get.ready else false.B

  val completeArb = Instantiate(new Arbiter(new Complete(frontendParam), if (gpParam.enable) 3 else 2))

  val completeQueue = Instantiate(new Queue(
    UInt(log2Up(frontendParam.rob_entries).W),
    frontendParam.rob_entries,
    false,
    false
  ))

  rob.io.complete <> completeQueue.io.deq
  completeArb.io.in(0).valid       := io.ballCompletion.valid && io.ballCompletion.bits.fault.error === 0.U
  completeArb.io.in(0).bits.rob_id := io.ballCompletion.bits.rob_id
  io.ballCompletion.ready          := completeArb.io.in(0).ready || io.ballCompletion.bits.fault.error =/= 0.U
  completeArb.io.in(1).valid       := io.memoryCompletion.valid && !memoryFailed
  completeArb.io.in(1).bits.rob_id := io.memoryCompletion.bits.rob_id
  io.memoryCompletion.ready        := memoryFailed || completeArb.io.in(1).ready

  if (gpParam.enable) { completeArb.io.in(2) <> io.gpCompletion.get }

  val completeBits = completeArb.io.out.bits

  io.ballRequest.valid            := rob.io.issue.valid && is_ball_domain && !failed
  io.ballRequest.bits.operation   := mainIssueEntry
  io.ballRequest.bits.context     := rob.io.issue.bits.context
  io.memoryRequest.valid          := rob.io.issue.valid && is_mem_domain && !failed
  io.memoryRequest.bits.operation := mainIssueEntry
  io.memoryRequest.bits.context   := rob.io.issue.bits.context
  rob.io.issue.ready              := !failed && ((is_ball_domain && io.ballRequest.ready) ||
    (is_mem_domain && io.memoryRequest.ready) || (is_kernel_domain && kernelReady))

  completeQueue.io.enq.valid := completeArb.io.out.valid
  completeQueue.io.enq.bits  := completeBits.rob_id
  completeArb.io.out.ready   := completeQueue.io.enq.ready
  io.admission.idle          := rob.io.empty && !barrierWaitROB && !barrierWaitRelease && !failed

  // Reset-time initialization is not a runtime command. Monitor runtime blocking locally.
  val runtimeStarted = RegInit(false.B)
  when(io.admission.command.fire && io.admission.command.bits.operation.runtime)(runtimeStarted := true.B)
  val blocked       = rob.io.full || barrierWaitROB || barrierWaitRelease
  val blockedCycles = RegInit(0.U(32.W))
  blockedCycles := Mux(runtimeStarted && blocked, blockedCycles + 1.U, 0.U)
  assert(blockedCycles < 10000000.U, "Accelerator: busy for too long!")

}
