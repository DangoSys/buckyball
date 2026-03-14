package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.decoder.GISA._
import framework.frontend.scoreboard.{BankAccessInfo, BankScoreboard}
import framework.core.bbtile.RoCCResponseBB
import framework.balldomain.blink.SubRobRow

// Global ROB entry - contains instruction + bank access info
class GlobalRobEntry(val b: GlobalConfig) extends Bundle {
  val cmd    = new PostGDCmd(b)
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Global RS issue interface
class GlobalRsIssue(b: GlobalConfig) extends GlobalRobEntry(b) {
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

// Global RS completion interface
class GlobalRsComplete(b: GlobalConfig) extends Bundle {
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

// Global reservation station - between GlobalDecoder and each Domain
@instantiable
class GlobalReservationStation(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // GlobalDecoder -> Global RS
    val global_decode_cmd_i = Flipped(new DecoupledIO(new PostGDCmd(b)))
    // Global RS -> BallDomain
    //           -> MemDomain
    //           -> GpDomain
    val ball_issue_o        = Decoupled(new GlobalRsIssue(b))
    val mem_issue_o         = Decoupled(new GlobalRsIssue(b))
    val gp_issue_o          = Decoupled(new GlobalRsIssue(b))
    // BallDomain -> Global RS
    // MemDomain  ->
    // GpDomain   ->
    val ball_complete_i     = Flipped(Decoupled(new GlobalRsComplete(b)))
    val mem_complete_i      = Flipped(Decoupled(new GlobalRsComplete(b)))
    val gp_complete_i       = Flipped(Decoupled(new GlobalRsComplete(b)))

    // Ball -> SubROB: write a row of sub-instructions
    val ball_subrob_req_i = Flipped(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))

    // RoCC response
    val rs_rocc_o = new Bundle {
      val resp = new DecoupledIO(new RoCCResponseBB(b.core.xLen))
      val busy = Output(Bool())
    }

    // Barrier interface — connected to tile-level BarrierUnit via BuckyballAccelerator
    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  val rob:    Instance[GlobalROB] = Instantiate(new GlobalROB(b))
  val subRob: Instance[SubROB]    = Instantiate(new SubROB(b))

// -----------------------------------------------------------------------------
// Fence handling — fence does NOT enter ROB.
// When a fence arrives, hold busy (stop CPU) until ROB drains completely.
// -----------------------------------------------------------------------------
  val isFenceCmd = io.global_decode_cmd_i.valid && io.global_decode_cmd_i.bits.isFence

  // Fence active state: set when fence arrives, cleared when ROB is empty
  val fenceActive = RegInit(false.B)
  when(isFenceCmd && !fenceActive) {
    fenceActive := true.B
  }
  when(fenceActive && rob.io.empty) {
    fenceActive := false.B
  }

// -----------------------------------------------------------------------------
// Barrier handling — barrier does NOT enter ROB.
// Two phases:
//   1. barrierWaitROB: wait for own ROB to drain (implicit fence)
//   2. barrierWaitRelease: arrive at BarrierUnit, wait for all cores
// -----------------------------------------------------------------------------
  val isBarrierCmd = io.global_decode_cmd_i.valid && io.global_decode_cmd_i.bits.isBarrier

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

// -----------------------------------------------------------------------------
// Inbound - instruction allocation (fence/barrier do not enter ROB)
// -----------------------------------------------------------------------------
  val isFrontendCmd = io.global_decode_cmd_i.bits.isFence || io.global_decode_cmd_i.bits.isBarrier
  val anyStall      = fenceActive || barrierWaitROB || barrierWaitRelease

  rob.io.alloc.valid := io.global_decode_cmd_i.valid && !isFrontendCmd && !anyStall
  rob.io.alloc.bits  := io.global_decode_cmd_i.bits

  // Backpressure: fence/barrier are consumed immediately (if not already active),
  // normal cmds wait for ROB ready and no stall.
  io.global_decode_cmd_i.ready := Mux(
    isFrontendCmd,
    !anyStall,                      // Accept fence/barrier if no stall active
    rob.io.alloc.ready && !anyStall // Normal cmd: ROB ready and no stall
  )

// -----------------------------------------------------------------------------
// SubROB write arbiter
// -----------------------------------------------------------------------------
  val subRobWriteArb = Module(new Arbiter(new SubRobRow(b), b.ballDomain.ballNum))
  for (i <- 0 until b.ballDomain.ballNum) {
    subRobWriteArb.io.in(i) <> io.ball_subrob_req_i(i)
  }
  subRob.io.write <> subRobWriteArb.io.out

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch to corresponding domain based on domain_id)
// SubROB issues take priority over main ROB issues.
// -----------------------------------------------------------------------------
  val is_ball_domain = rob.io.issue.bits.cmd.domain_id === DomainId.BALL
  val is_mem_domain  = rob.io.issue.bits.cmd.domain_id === DomainId.MEM
  val is_gp_domain   = rob.io.issue.bits.cmd.domain_id === DomainId.GP

  val subRobIssueValid = subRob.io.issue.valid
  val subRobCmd        = subRob.io.issue.bits // PostGDCmd

  // Build a GlobalRsIssue for the sub-instruction
  val subRobIssueEntry = Wire(new GlobalRsIssue(b))
  subRobIssueEntry.cmd        := subRobCmd
  subRobIssueEntry.rob_id     := subRob.io.issueMasterRobId
  subRobIssueEntry.is_sub     := true.B
  subRobIssueEntry.sub_rob_id := subRob.io.issueSubId

  val subRobIssBall = subRobCmd.domain_id === DomainId.BALL
  val subRobIssMem  = subRobCmd.domain_id === DomainId.MEM
  val subRobIssGp   = subRobCmd.domain_id === DomainId.GP

  // Build main ROB issue entry with is_sub/sub_rob_id cleared
  val mainIssueEntry = Wire(new GlobalRsIssue(b))
  mainIssueEntry.cmd        := rob.io.issue.bits.cmd
  mainIssueEntry.rob_id     := rob.io.issue.bits.rob_id
  mainIssueEntry.is_sub     := false.B
  mainIssueEntry.sub_rob_id := 0.U

  // Ball issue: SubROB priority
  io.ball_issue_o.valid := Mux(
    subRobIssueValid && subRobIssBall,
    true.B,
    rob.io.issue.valid && is_ball_domain && !subRobIssueValid
  )
  io.ball_issue_o.bits  := Mux(subRobIssueValid && subRobIssBall, subRobIssueEntry, mainIssueEntry)

  // Mem issue: SubROB priority
  io.mem_issue_o.valid := Mux(
    subRobIssueValid && subRobIssMem,
    true.B,
    rob.io.issue.valid && is_mem_domain && !subRobIssueValid
  )
  io.mem_issue_o.bits  := Mux(subRobIssueValid && subRobIssMem, subRobIssueEntry, mainIssueEntry)

  // GP issue: SubROB priority
  io.gp_issue_o.valid := Mux(
    subRobIssueValid && subRobIssGp,
    true.B,
    rob.io.issue.valid && is_gp_domain && !subRobIssueValid
  )
  io.gp_issue_o.bits  := Mux(subRobIssueValid && subRobIssGp, subRobIssueEntry, mainIssueEntry)

  // SubROB issue ready
  subRob.io.issue.ready :=
    (subRobIssBall && io.ball_issue_o.ready) ||
      (subRobIssMem && io.mem_issue_o.ready) ||
      (subRobIssGp && io.gp_issue_o.ready)

  // Main ROB issue ready: yield when SubROB is issuing
  rob.io.issue.ready := !subRobIssueValid && (
    (is_ball_domain && io.ball_issue_o.ready) ||
      (is_mem_domain && io.mem_issue_o.ready) ||
      (is_gp_domain && io.gp_issue_o.ready)
  )

  // Tell GlobalROB to suppress its own issue when SubROB is active
  rob.io.subRobActive := subRobIssueValid

// -----------------------------------------------------------------------------
// Completion signal processing
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(new GlobalRsComplete(b), 3))

  // Connect Ball, Mem, and GP domain completion signals to arbiter
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits
  io.ball_complete_i.ready   := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits
  io.mem_complete_i.ready    := completeArb.io.in(1).ready

  completeArb.io.in(2).valid := io.gp_complete_i.valid
  completeArb.io.in(2).bits  := io.gp_complete_i.bits
  io.gp_complete_i.ready     := completeArb.io.in(2).ready

  val completeBits = completeArb.io.out.bits

  // Route sub-completions to SubROB, main completions to main ROB
  subRob.io.subComplete.valid := completeArb.io.out.valid && completeBits.is_sub
  subRob.io.subComplete.bits  := completeBits.sub_rob_id

  subRob.io.masterComplete.ready := true.B // main ROB always ready to accept masterComplete

  val normalComplete = completeArb.io.out.valid && !completeBits.is_sub

  if (b.frontend.rs_out_of_order_response) {
    rob.io.complete.valid := normalComplete || subRob.io.masterComplete.valid
    rob.io.complete.bits  := Mux(subRob.io.masterComplete.valid, subRob.io.masterComplete.bits, completeBits.rob_id)
  } else {
    val isHeadComplete = Mux(
      subRob.io.masterComplete.valid,
      subRob.io.masterComplete.bits === rob.io.head_ptr,
      completeBits.rob_id === rob.io.head_ptr
    )
    rob.io.complete.valid := (normalComplete || subRob.io.masterComplete.valid) && isHeadComplete
    rob.io.complete.bits  := Mux(subRob.io.masterComplete.valid, subRob.io.masterComplete.bits, completeBits.rob_id)
  }

  completeArb.io.out.ready := Mux(
    completeBits.is_sub,
    subRob.io.subComplete.ready,
    rob.io.complete.ready
  )

// -----------------------------------------------------------------------------
// Response generation
// -----------------------------------------------------------------------------
  io.rs_rocc_o.resp.valid     := false.B
  io.rs_rocc_o.resp.bits.rd   := 0.U
  io.rs_rocc_o.resp.bits.data := 0.U
  // busy when ROB is not empty OR fence/barrier is active
  io.rs_rocc_o.busy           := !rob.io.empty || fenceActive || barrierWaitROB || barrierWaitRelease
}
