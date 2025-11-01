package framework.builtin.frontend.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain._
import framework.rocket.RoCCResponseBB
import framework.blink.BallRegist

// Ball device information - configuration information for registering Ball devices
case class BallRsRegist(
  ballId: Int,
  ballName: String
)

// Ball domain issue interface - includes global rob_id
class BallRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd = new BallDecodeCmd
  // Global ROB ID
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// Ball domain completion interface
class BallRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// Generic Ball domain issue interface - supports dynamic number of Ball devices
class BallIssueInterface(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val balls = Vec(numBalls, Decoupled(new BallRsIssue))
}

// Generic Ball domain completion interface - supports dynamic number of Ball devices
class BallCommitInterface(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val balls = Vec(numBalls, Flipped(Decoupled(new BallRsComplete)))
}

// Local Ball reservation station - simple FIFO scheduler
class BallReservationStation(BallRsRegists: Seq[BallRsRegist])
  (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {

  val numBalls = BallRsRegists.length

  val io = IO(new Bundle {
    // Decoded instruction input (with global rob_id)
    val ball_decode_cmd_i = Flipped(new DecoupledIO(new Bundle {
      val cmd = new BallDecodeCmd
      // Global ROB ID
      val rob_id = UInt(log2Up(b.rob_entries).W)
    }))

    // Rs -> BallController (multi-channel issue)
    val issue_o     = new BallIssueInterface(numBalls)
    val commit_i    = new BallCommitInterface(numBalls)

    // Output completion signal (with global rob_id, single channel)
    val complete_o = Decoupled(new BallRsComplete)
  })

  // Simple FIFO queue, only for buffering
  val fifo = Module(new Queue(new Bundle {
    val cmd = new BallDecodeCmd
    val rob_id = UInt(log2Up(b.rob_entries).W)
  }, entries = 4))  // Small buffer is sufficient

// -----------------------------------------------------------------------------
// Inbound - FIFO enqueue
// -----------------------------------------------------------------------------
  fifo.io.enq <> io.ball_decode_cmd_i

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch to corresponding Ball device based on bid)
// -----------------------------------------------------------------------------
  val headEntry = fifo.io.deq.bits

  // Set issue signals for each Ball device
  for (i <- 0 until numBalls) {
    val ballId = BallRsRegists(i).ballId.U
    io.issue_o.balls(i).valid := fifo.io.deq.valid && headEntry.cmd.bid === ballId
    io.issue_o.balls(i).bits.cmd := headEntry.cmd
    io.issue_o.balls(i).bits.rob_id := headEntry.rob_id
  }

  // FIFO deq.ready - can only dequeue when target Ball device is ready
  fifo.io.deq.ready := VecInit(
    BallRsRegists.zipWithIndex.map { case (info, idx) =>
      (headEntry.cmd.bid === info.ballId.U) && io.issue_o.balls(idx).ready
    }
  ).asUInt.orR

// -----------------------------------------------------------------------------
// Completion signal processing - directly forward to global RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), numBalls))

  // Connect completion signals from all Ball devices to arbiter
  for (i <- 0 until numBalls) {
    completeArb.io.in(i).valid := io.commit_i.balls(i).valid
    completeArb.io.in(i).bits  := io.commit_i.balls(i).bits.rob_id
    io.commit_i.balls(i).ready := completeArb.io.in(i).ready
  }

  // Forward completion signal (with global rob_id)
  io.complete_o.valid := completeArb.io.out.valid
  io.complete_o.bits.rob_id := completeArb.io.out.bits
  completeArb.io.out.ready := io.complete_o.ready
}
