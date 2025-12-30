package framework.balldomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import examples.toy.balldomain.BallDecodeCmd

// Ball domain issue interface - includes global rob_id
class BallRsIssue(b: GlobalConfig) extends Bundle {
  val cmd    = new BallDecodeCmd(b.memDomain.bankNum)
  // Global ROB ID
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Ball domain completion interface
class BallRsComplete(b: GlobalConfig) extends Bundle {
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Generic Ball domain issue interface - supports dynamic number of Ball devices
class BallIssueInterface(b: GlobalConfig) extends Bundle {
  val balls = Vec(b.ballDomain.ballNum, Decoupled(new BallRsIssue(b)))
}

// Generic Ball domain completion interface - supports dynamic number of Ball devices
class BallCommitInterface(b: GlobalConfig) extends Bundle {
  val balls = Vec(b.ballDomain.ballNum, Flipped(Decoupled(new BallRsComplete(b))))
}

// Local Ball reservation station - simple FIFO scheduler
@instantiable
class BallReservationStation(val b: GlobalConfig) extends Module {

  @public
  val ball_decode_cmd_i = IO(Flipped(new DecoupledIO(new Bundle {
    val cmd    = new BallDecodeCmd(b.memDomain.bankNum)
    // Global ROB ID
    val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
  })))

  // Rs -> BallController (multi-channel issue)
  @public
  val issue_o = IO(new BallIssueInterface(b))

  @public
  val commit_i = IO(new BallCommitInterface(b))

  // Output completion signal (with global rob_id, single channel)
  @public
  val complete_o = IO(Decoupled(new BallRsComplete(b)))

  // Simple FIFO queue, only for buffering
  val fifo = Module(new Queue(
    new Bundle {
      val cmd    = new BallDecodeCmd(b.memDomain.bankNum)
      val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
    },
    entries = 4
  )) // Small buffer is sufficient

// -----------------------------------------------------------------------------
// Inbound - FIFO enqueue
// -----------------------------------------------------------------------------
  fifo.io.enq <> ball_decode_cmd_i

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch to corresponding Ball device based on bid)
// -----------------------------------------------------------------------------
  val headEntry = fifo.io.deq.bits

  // Build ballId to index mapping from config
  // Config order should match the order in busRegister.scala
  // Each index in issue_o.balls corresponds to a ball registered in BBus
  val numConfiguredBalls = b.ballDomain.ballIdMappings.length

  // Set issue signals for each Ball device
  // Use configured ball id mappings: index i in issue_o.balls corresponds to ballIdMappings(i).ballId
  for (i <- 0 until b.ballDomain.ballNum) {
    if (i < numConfiguredBalls) {
      val configuredBallId = b.ballDomain.ballIdMappings(i).ballId.U
      issue_o.balls(i).valid       := fifo.io.deq.valid && headEntry.cmd.bid === configuredBallId
      issue_o.balls(i).bits.cmd    := headEntry.cmd
      issue_o.balls(i).bits.rob_id := headEntry.rob_id
    } else {
      // Unused slots - no valid signal
      issue_o.balls(i).valid       := false.B
      issue_o.balls(i).bits.cmd    := DontCare
      issue_o.balls(i).bits.rob_id := DontCare
    }
  }

  // FIFO deq.ready - can only dequeue when target Ball device is ready
  // Find which index corresponds to the requested ballId
  fifo.io.deq.ready := VecInit(
    (0 until b.ballDomain.ballNum).map { idx =>
      if (idx < numConfiguredBalls) {
        val configuredBallId = b.ballDomain.ballIdMappings(idx).ballId.U
        (headEntry.cmd.bid === configuredBallId) && issue_o.balls(idx).ready
      } else {
        false.B
      }
    }
  ).asUInt.orR

// -----------------------------------------------------------------------------
// Completion signal processing - directly forward to global RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.frontend.rob_entries).W), b.ballDomain.ballNum))

  // Connect completion signals from all Ball devices to arbiter
  for (i <- 0 until b.ballDomain.ballNum) {
    completeArb.io.in(i).valid := commit_i.balls(i).valid
    completeArb.io.in(i).bits  := commit_i.balls(i).bits.rob_id
    commit_i.balls(i).ready    := completeArb.io.in(i).ready
  }

  // Forward completion signal (with global rob_id)
  complete_o.valid         := completeArb.io.out.valid
  complete_o.bits.rob_id   := completeArb.io.out.bits
  completeArb.io.out.ready := complete_o.ready
}
