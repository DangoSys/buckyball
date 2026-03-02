package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.decoder.GISA._
import framework.frontend.scoreboard.{BankAccessInfo, BankScoreboard}
import framework.core.rocket.RoCCResponseBB

// Global ROB entry - contains instruction + bank access info
class GlobalRobEntry(val b: GlobalConfig) extends Bundle {
  val cmd    = new PostGDCmd(b)
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Global RS issue interface
class GlobalRsIssue(b: GlobalConfig) extends GlobalRobEntry(b)

// Global RS completion interface
class GlobalRsComplete(b: GlobalConfig) extends Bundle {
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
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

    // RoCC response
    val rs_rocc_o = new Bundle {
      val resp = new DecoupledIO(new RoCCResponseBB(b.core.xLen))
      val busy = Output(Bool())
    }

  })

  val rob: Instance[GlobalROB] = Instantiate(new GlobalROB(b))

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
// Inbound - instruction allocation (only non-fence instructions enter ROB)
// -----------------------------------------------------------------------------
  rob.io.alloc.valid := io.global_decode_cmd_i.valid && !io.global_decode_cmd_i.bits.isFence && !fenceActive
  rob.io.alloc.bits  := io.global_decode_cmd_i.bits

  // Backpressure: during fence, accept the fence cmd immediately but block further cmds
  // until ROB drains. For normal cmds, wait for ROB ready.
  io.global_decode_cmd_i.ready := Mux(
    io.global_decode_cmd_i.bits.isFence,
    !fenceActive,                      // Accept fence if not already processing one
    rob.io.alloc.ready && !fenceActive // Normal cmd: ROB ready and no fence active
  )

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch to corresponding domain based on domain_id)
// -----------------------------------------------------------------------------
  val is_ball_domain = rob.io.issue.bits.cmd.domain_id === DomainId.BALL
  val is_mem_domain  = rob.io.issue.bits.cmd.domain_id === DomainId.MEM
  val is_gp_domain   = rob.io.issue.bits.cmd.domain_id === DomainId.GP

  // Ball domain issue
  io.ball_issue_o.valid := rob.io.issue.valid && is_ball_domain
  io.ball_issue_o.bits  := rob.io.issue.bits

  // Mem domain issue
  io.mem_issue_o.valid := rob.io.issue.valid && is_mem_domain
  io.mem_issue_o.bits  := rob.io.issue.bits

  // GP domain issue
  io.gp_issue_o.valid := rob.io.issue.valid && is_gp_domain
  io.gp_issue_o.bits  := rob.io.issue.bits

  // Set ROB ready signal - can only issue when target domain is ready
  rob.io.issue.ready :=
    (is_ball_domain && io.ball_issue_o.ready) ||
      (is_mem_domain && io.mem_issue_o.ready) ||
      (is_gp_domain && io.gp_issue_o.ready)

// -----------------------------------------------------------------------------
// Completion signal processing
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.frontend.rob_entries).W), 3))

  // Connect Ball, Mem, and GP domain completion signals to arbiter
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits.rob_id
  io.ball_complete_i.ready   := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits.rob_id
  io.mem_complete_i.ready    := completeArb.io.in(1).ready

  completeArb.io.in(2).valid := io.gp_complete_i.valid
  completeArb.io.in(2).bits  := io.gp_complete_i.bits.rob_id
  io.gp_complete_i.ready     := completeArb.io.in(2).ready

  // Decide whether to filter completion signals based on configuration
  if (b.frontend.rs_out_of_order_response) {
    // Out-of-order mode: accept all completion signals, ROB commits out-of-order internally
    rob.io.complete <> completeArb.io.out
  } else {
    // Sequential mode: only accept completion signals where rob_id == head_ptr
    val isHeadComplete = completeArb.io.out.bits === rob.io.head_ptr
    rob.io.complete.valid    := completeArb.io.out.valid && isHeadComplete
    rob.io.complete.bits     := completeArb.io.out.bits
    completeArb.io.out.ready := rob.io.complete.ready && isHeadComplete
  }

// -----------------------------------------------------------------------------
// Response generation
// -----------------------------------------------------------------------------
  io.rs_rocc_o.resp.valid     := false.B
  io.rs_rocc_o.resp.bits.rd   := 0.U
  io.rs_rocc_o.resp.bits.data := 0.U
  // busy when ROB is not empty OR fence is active (waiting for drain)
  io.rs_rocc_o.busy           := !rob.io.empty || fenceActive
}
