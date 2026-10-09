package framework.frontend.globalrs

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.decoder.GISA._
import framework.frontend.scoreboard.BankAccessInfo
import framework.system.core.rocket.RoCCResponseBB
import framework.balldomain.blink.SubRobRow
import framework.memdomain.backend.banks.btrace.{BTraceDPI, PhysicalBankHash}
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.mem.dma.DmaStatus

class GlobalRobEntry(val b: GlobalConfig) extends Bundle {
  val cmd               = new PostGDCmd(b)
  val renamedBankAccess = new BankAccessInfo(b.frontend.bank_id_len)
  val rob_id            = UInt(log2Up(b.frontend.rob_entries).W)
}

class GlobalSchedIssue(b: GlobalConfig) extends GlobalRobEntry(b) {
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

class GlobalSchedComplete(b: GlobalConfig) extends Bundle {
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
  val fault      = new DmaStatus
}

class RobAllocation(b: GlobalConfig) extends Bundle {
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

@instantiable
class GlobalScheduler(val b: GlobalConfig) extends Module {

  val sharedHashCount = if (b.memDomain.sharedEnable) SharedMemLayout.totalBank(b) else 0

  @public
  val io = IO(new Bundle {
    val hart_id               = Input(UInt(b.tile.xLen.W))
    val sharedBankOwnerHartId = Input(UInt(b.tile.xLen.W))
    val decode_cmd_i          = Flipped(new DecoupledIO(new PostGDCmd(b)))
    val ball_issue_o          = Decoupled(new GlobalSchedIssue(b))
    val mem_issue_o           = Decoupled(new GlobalSchedIssue(b))
    val ball_complete_i       = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val mem_complete_i        = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val kernel_issue_o        = if (b.rvv.enable) Some(Decoupled(new GlobalSchedIssue(b))) else None
    val kernel_complete_i     = if (b.rvv.enable) Some(Flipped(Decoupled(new GlobalSchedComplete(b)))) else None
    val ball_subrob_req_i     = Flipped(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))
    val inst_ids              = Output(Vec(b.frontend.rob_entries, UInt(64.W)))
    val allocation            = Valid(new RobAllocation(b))
    val retired               = Output(UInt(b.frontend.rob_entries.W))

    val bank_hashes =
      if (b.sim.diffTest) {
        Some(Input(Vec(b.memDomain.bankNum + sharedHashCount, new PhysicalBankHash(b))))
      } else {
        None
      }

    val scheduler_rocc_o = new Bundle {
      val resp = new DecoupledIO(new RoCCResponseBB(b.tile.xLen))
      val busy = Output(Bool())
    }

    val idle = Output(Bool())

    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  val rob: Instance[GlobalROB] = Instantiate(new GlobalROB(b))
  rob.io.hart_id               := io.hart_id
  rob.io.bank_hashes.foreach(_ := io.bank_hashes.get)
  io.inst_ids                  := rob.io.inst_ids
  io.allocation                := rob.io.allocation
  io.retired                   := rob.io.retired

  if (b.sim.diffTest) {
    val btrace = Module(new BTraceDPI)
    val trace  = rob.io.trace.get
    // VVAC stalls logical clocks when its nonblocking channel is full.
    btrace.io.clock  := clock
    btrace.io.reset  := reset.asBool
    btrace.io.instId := trace.bits.instId
    btrace.io.hartId := io.hart_id
    val sharedBank = trace.bits.w0Vbank > b.frontend.vbank_id_upper_bound.U &&
      trace.bits.w0Vbank >= b.frontend.shared_bank_id_base.U &&
      trace.bits.w0Vbank < b.memDomain.virtualBankCount.U
    btrace.io.ownerHartId := Mux(sharedBank, io.sharedBankOwnerHartId, io.hart_id)
    btrace.io.w0Vbank     := trace.bits.w0Vbank
    btrace.io.w0Hash      := trace.bits.w0Hash
    btrace.io.fire        := trace.valid
    btrace.io.produced    := rob.io.traceCount.get
    btrace.io.idle        := io.idle && !io.decode_cmd_i.valid && !trace.valid
  }

  val isFenceCmd  = io.decode_cmd_i.valid && io.decode_cmd_i.bits.isFence
  val fenceActive = RegInit(false.B)
  when(isFenceCmd && !fenceActive) {
    fenceActive := true.B
  }
  when(fenceActive && rob.io.empty) {
    fenceActive := false.B
  }

  val isBarrierCmd       = io.decode_cmd_i.valid && io.decode_cmd_i.bits.isBarrier
  val barrierWaitROB     = RegInit(false.B)
  val barrierWaitRelease = RegInit(false.B)
  when(isBarrierCmd && !barrierWaitROB && !barrierWaitRelease && !fenceActive) {
    barrierWaitROB := true.B
  }
  when(barrierWaitROB && rob.io.empty) {
    barrierWaitROB     := false.B
    barrierWaitRelease := true.B
  }
  when(barrierWaitRelease && io.barrier_release) {
    barrierWaitRelease := false.B
  }
  io.barrier_arrive := barrierWaitRelease

  val isFrontendCmd = io.decode_cmd_i.bits.isFence || io.decode_cmd_i.bits.isBarrier
  val anyStall      = fenceActive || barrierWaitROB || barrierWaitRelease
  rob.io.alloc.valid    := io.decode_cmd_i.valid && !isFrontendCmd && !anyStall
  rob.io.alloc.bits     := io.decode_cmd_i.bits
  io.decode_cmd_i.ready := Mux(
    isFrontendCmd,
    !anyStall,
    rob.io.alloc.ready && !anyStall
  )

  val is_ball_domain   = rob.io.issue.bits.cmd.domain_id === DomainId.BALL
  val is_mem_domain    = rob.io.issue.bits.cmd.domain_id === DomainId.MEM
  val is_kernel_domain = rob.io.issue.bits.cmd.domain_id === DomainId.RVV

  val mainIssueEntry = Wire(new GlobalSchedIssue(b))
  mainIssueEntry.cmd               := rob.io.issue.bits.cmd
  mainIssueEntry.renamedBankAccess := rob.io.issue.bits.renamedBankAccess
  mainIssueEntry.rob_id            := rob.io.issue.bits.rob_id
  mainIssueEntry.is_sub            := false.B
  mainIssueEntry.sub_rob_id        := 0.U

  if (b.rvv.enable) {
    io.kernel_issue_o.get.valid := rob.io.issue.valid && is_kernel_domain
    io.kernel_issue_o.get.bits  := mainIssueEntry
  }
  val kernelReady = if (b.rvv.enable) io.kernel_issue_o.get.ready else false.B

  val completeArb   = Module(new Arbiter(new GlobalSchedComplete(b), if (b.rvv.enable) 3 else 2))
  val completeQueue = Module(new Queue(UInt(log2Up(b.frontend.rob_entries).W), b.frontend.rob_entries))
  rob.io.complete <> completeQueue.io.deq
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits
  io.ball_complete_i.ready   := completeArb.io.in(0).ready
  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits
  io.mem_complete_i.ready    := completeArb.io.in(1).ready

  if (b.rvv.enable) { completeArb.io.in(2) <> io.kernel_complete_i.get }

  val completeBits = completeArb.io.out.bits

  if (b.frontend.sub_rob_enable) {
    val subRob: Instance[SubROB] = Instantiate(new SubROB(b))

    val subRobWriteArb = Module(new Arbiter(new SubRobRow(b), b.ballDomain.ballNum))
    for (i <- 0 until b.ballDomain.ballNum) {
      subRobWriteArb.io.in(i) <> io.ball_subrob_req_i(i)
    }
    subRob.io.write <> subRobWriteArb.io.out

    val subRobIssueValid = subRob.io.issue.valid
    val subRobCmd        = subRob.io.issue.bits

    val subRobIssueEntry = Wire(new GlobalSchedIssue(b))
    subRobIssueEntry.cmd               := subRobCmd
    subRobIssueEntry.renamedBankAccess := 0.U.asTypeOf(subRobIssueEntry.renamedBankAccess)
    subRobIssueEntry.rob_id            := subRob.io.issueMasterRobId
    subRobIssueEntry.is_sub            := true.B
    subRobIssueEntry.sub_rob_id        := subRob.io.issueSubId

    val subRobIssBall = subRobCmd.domain_id === DomainId.BALL
    val subRobIssMem  = subRobCmd.domain_id === DomainId.MEM
    when(subRobIssueValid) {
      assert(
        (subRobCmd.cmd.funct =/= MVIN_KERNEL_BITPAT) && (subRobCmd.cmd.funct =/= RUN_KERNEL_BITPAT),
        "RVV kernel commands cannot originate from SubROB"
      )
    }

    io.ball_issue_o.valid := Mux(
      subRobIssueValid && subRobIssBall,
      true.B,
      rob.io.issue.valid && is_ball_domain && !subRobIssueValid
    )
    io.ball_issue_o.bits  := Mux(subRobIssueValid && subRobIssBall, subRobIssueEntry, mainIssueEntry)

    io.mem_issue_o.valid := Mux(
      subRobIssueValid && subRobIssMem,
      true.B,
      rob.io.issue.valid && is_mem_domain && !subRobIssueValid
    )
    io.mem_issue_o.bits  := Mux(subRobIssueValid && subRobIssMem, subRobIssueEntry, mainIssueEntry)

    subRob.io.issue.ready :=
      (subRobIssBall && io.ball_issue_o.ready) ||
        (subRobIssMem && io.mem_issue_o.ready)

    if (b.rvv.enable) { io.kernel_issue_o.get.valid := rob.io.issue.valid && is_kernel_domain && !subRobIssueValid }
    rob.io.issue.ready  := !subRobIssueValid && (
      (is_ball_domain && io.ball_issue_o.ready) ||
        (is_mem_domain && io.mem_issue_o.ready) || (is_kernel_domain && kernelReady)
    )
    rob.io.subRobActive := subRobIssueValid

    subRob.io.subComplete.valid := completeArb.io.out.valid && completeBits.is_sub
    subRob.io.subComplete.bits  := completeBits.sub_rob_id

    val normalComplete = completeArb.io.out.valid && !completeBits.is_sub
    val masterComplete = subRob.io.masterComplete.valid
    val completeRobId  = Mux(masterComplete, subRob.io.masterComplete.bits, completeBits.rob_id)

    completeQueue.io.enq.valid     := masterComplete || normalComplete
    completeQueue.io.enq.bits      := completeRobId
    subRob.io.masterComplete.ready := masterComplete && completeQueue.io.enq.ready
    completeArb.io.out.ready       := Mux(
      completeBits.is_sub,
      subRob.io.subComplete.ready,
      !masterComplete && completeQueue.io.enq.ready
    )
    io.idle                        := rob.io.empty && !fenceActive && !barrierWaitROB && !barrierWaitRelease && !subRob.io.occupied
  } else {
    for (i <- 0 until b.ballDomain.ballNum) {
      io.ball_subrob_req_i(i).ready := false.B
    }

    io.ball_issue_o.valid := rob.io.issue.valid && is_ball_domain
    io.ball_issue_o.bits  := mainIssueEntry
    io.mem_issue_o.valid  := rob.io.issue.valid && is_mem_domain
    io.mem_issue_o.bits   := mainIssueEntry
    rob.io.issue.ready    := (is_ball_domain && io.ball_issue_o.ready) ||
      (is_mem_domain && io.mem_issue_o.ready) || (is_kernel_domain && kernelReady)
    rob.io.subRobActive   := false.B

    when(completeArb.io.out.valid) {
      assert(!completeBits.is_sub, "SubROB completion observed when frontend.sub_rob_enable=false")
    }
    completeQueue.io.enq.valid := completeArb.io.out.valid
    completeQueue.io.enq.bits  := completeBits.rob_id
    completeArb.io.out.ready   := completeQueue.io.enq.ready
    io.idle                    := rob.io.empty && !fenceActive && !barrierWaitROB && !barrierWaitRelease
  }

  io.scheduler_rocc_o.resp.valid     := false.B
  io.scheduler_rocc_o.resp.bits.rd   := 0.U
  io.scheduler_rocc_o.resp.bits.data := 0.U
  io.scheduler_rocc_o.busy           := rob.io.full || fenceActive || barrierWaitROB || barrierWaitRelease

}
