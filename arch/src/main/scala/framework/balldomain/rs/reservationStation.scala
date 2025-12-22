package framework.balldomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.toy.balldomain._
import framework.balldomain.blink.BallRegist

// Ball device information - configuration information for registering Ball devices
case class BallRsRegist(
  ballId:   Int,
  ballName: String)

// Ball domain issue interface - includes global rob_id
class BallRsIssue(parameter: BallDomainParam) extends Bundle {
  val cmd    = new BallDecodeCmd(parameter)
  // Global ROB ID
  val rob_id = UInt(log2Up(parameter.rob_entries).W)
}

// Ball domain completion interface
class BallRsComplete(parameter: BallDomainParam) extends Bundle {
  val rob_id = UInt(log2Up(parameter.rob_entries).W)
}

// Generic Ball domain issue interface - supports dynamic number of Ball devices
class BallIssueInterface(numBalls: Int, parameter: BallDomainParam) extends Bundle {
  val balls = Vec(numBalls, Decoupled(new BallRsIssue(parameter)))
}

// Generic Ball domain completion interface - supports dynamic number of Ball devices
class BallCommitInterface(numBalls: Int, parameter: BallDomainParam) extends Bundle {
  val balls = Vec(numBalls, Flipped(Decoupled(new BallRsComplete(parameter))))
}

// Local Ball reservation station - simple FIFO scheduler
@instantiable
class BallReservationStation(val parameter: BallDomainParam, BallRsRegists: Seq[BallRsRegist]) extends Module {

  val numBalls = BallRsRegists.length

  @public
  val ball_decode_cmd_i = IO(Flipped(new DecoupledIO(new Bundle {
    val cmd    = new BallDecodeCmd(parameter)
    // Global ROB ID
    val rob_id = UInt(log2Up(parameter.rob_entries).W)
  })))

  // Rs -> BallController (multi-channel issue)
  @public
  val issue_o = IO(new BallIssueInterface(numBalls, parameter))

  @public
  val commit_i = IO(new BallCommitInterface(numBalls, parameter))

  // Output completion signal (with global rob_id, single channel)
  @public
  val complete_o = IO(Decoupled(new BallRsComplete(parameter)))

  // Simple FIFO queue, only for buffering
  val fifo = Module(new Queue(
    new Bundle {
      val cmd    = new BallDecodeCmd(parameter)
      val rob_id = UInt(log2Up(parameter.rob_entries).W)
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

  // Set issue signals for each Ball device
  for (i <- 0 until numBalls) {
    val ballId = BallRsRegists(i).ballId.U
    issue_o.balls(i).valid       := fifo.io.deq.valid && headEntry.cmd.bid === ballId
    issue_o.balls(i).bits.cmd    := headEntry.cmd
    issue_o.balls(i).bits.rob_id := headEntry.rob_id
  }

  // FIFO deq.ready - can only dequeue when target Ball device is ready
  fifo.io.deq.ready := VecInit(
    BallRsRegists.zipWithIndex.map {
      case (info, idx) =>
        (headEntry.cmd.bid === info.ballId.U) && issue_o.balls(idx).ready
    }
  ).asUInt.orR

// -----------------------------------------------------------------------------
// Completion signal processing - directly forward to global RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(parameter.rob_entries).W), numBalls))

  // Connect completion signals from all Ball devices to arbiter
  for (i <- 0 until numBalls) {
    completeArb.io.in(i).valid := commit_i.balls(i).valid
    completeArb.io.in(i).bits  := commit_i.balls(i).bits.rob_id
    commit_i.balls(i).ready    := completeArb.io.in(i).ready
  }

  // Forward completion signal (with global rob_id)
  complete_o.valid         := completeArb.io.out.valid
  complete_o.bits.rob_id   := completeArb.io.out.bits
  completeArb.io.out.ready := complete_o.ready
}
