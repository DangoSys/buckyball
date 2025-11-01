package framework.builtin.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.PostGDCmd
import framework.rocket.RoCCResponseBB

// Global ROB entry - only contains basic information, does not include specific instruction decoding
class GlobalRobEntry(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd    = new PostGDCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// Global RS issue interface
class GlobalRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends GlobalRobEntry

// Global RS completion interface
class GlobalRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// No additional interface Bundle needed, defined directly in IO

// Global reservation station - between GlobalDecoder and each Domain
class GlobalReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // GlobalDecoder -> Global RS
    val global_decode_cmd_i = Flipped(new DecoupledIO(new PostGDCmd))

    // Global RS -> BallDomain (single channel)
    val ball_issue_o = Decoupled(new GlobalRsIssue)

    // Global RS -> MemDomain (single channel)
    val mem_issue_o = Decoupled(new GlobalRsIssue)

    // BallDomain -> Global RS (single channel)
    val ball_complete_i = Flipped(Decoupled(new GlobalRsComplete))

    // MemDomain -> Global RS (single channel)
    val mem_complete_i = Flipped(Decoupled(new GlobalRsComplete))

    // RoCC response
    val rs_rocc_o = new Bundle {
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
  })

  val rob = Module(new GlobalROB)

// -----------------------------------------------------------------------------
// Fence handling
// -----------------------------------------------------------------------------
  val fenceActive = RegInit(false.B)
  // Cannot use fire, would form a loop
  val isFenceCmd = io.global_decode_cmd_i.valid && io.global_decode_cmd_i.bits.is_fence
  val robEmpty = rob.io.empty

  // Fence state machine: only activate when fence instruction is accepted (fire)
  when (io.global_decode_cmd_i.fire && isFenceCmd && !fenceActive) {
    fenceActive := true.B
  }
  when (fenceActive && robEmpty) {
    fenceActive := false.B
  }

// -----------------------------------------------------------------------------
// Inbound - instruction allocation (Fence instructions do not enter ROB)
// -----------------------------------------------------------------------------
  // Filter out fence instructions
  rob.io.alloc.valid := io.global_decode_cmd_i.valid && !isFenceCmd
  rob.io.alloc.bits  := io.global_decode_cmd_i.bits

  // Backpressure logic:
  // - Normal instructions: wait for ROB ready
  // - Fence instructions: wait for ROB empty
  io.global_decode_cmd_i.ready := Mux(isFenceCmd, robEmpty, rob.io.alloc.ready)

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch to corresponding domain based on is_ball/is_mem)
// -----------------------------------------------------------------------------
  // Ball domain issue
  io.ball_issue_o.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_ball
  io.ball_issue_o.bits  := rob.io.issue.bits

  // Mem domain issue
  io.mem_issue_o.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_mem
  io.mem_issue_o.bits  := rob.io.issue.bits

  // Set ROB ready signal - can only issue when target domain is ready
  rob.io.issue.ready :=
    (rob.io.issue.bits.cmd.is_ball && io.ball_issue_o.ready) ||
    (rob.io.issue.bits.cmd.is_mem  && io.mem_issue_o.ready)

// -----------------------------------------------------------------------------
// Completion signal processing
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))

  // Connect Ball and Mem domain completion signals to arbiter
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits.rob_id
  io.ball_complete_i.ready := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits.rob_id
  io.mem_complete_i.ready := completeArb.io.in(1).ready

  // Decide whether to filter completion signals based on configuration
  if (b.rs_out_of_order_response) {
    // Out-of-order mode: accept all completion signals, ROB commits out-of-order internally
    rob.io.complete <> completeArb.io.out
  } else {
    // Sequential mode: only accept completion signals where rob_id == head_ptr
    val isHeadComplete = completeArb.io.out.bits === rob.io.head_ptr
    rob.io.complete.valid := completeArb.io.out.valid && isHeadComplete
    rob.io.complete.bits  := completeArb.io.out.bits
    completeArb.io.out.ready := rob.io.complete.ready && isHeadComplete
  }

// -----------------------------------------------------------------------------
// Response generation
// -----------------------------------------------------------------------------
  // Response logic:
  // - Normal instructions: respond immediately
  // - Fence instructions: respond after ROB is empty
  io.rs_rocc_o.resp.valid := io.global_decode_cmd_i.fire &&
                             (!io.global_decode_cmd_i.bits.is_fence || robEmpty)
  io.rs_rocc_o.resp.bits  := DontCare
  io.rs_rocc_o.busy       := !rob.io.empty || fenceActive
}
