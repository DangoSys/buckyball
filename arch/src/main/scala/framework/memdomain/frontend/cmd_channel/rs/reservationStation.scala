package framework.memdomain.frontend.cmd_channel.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import framework.memdomain.frontend.cmd_channel.decoder.MemDecodeCmd
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

// Mem domain issue interface - includes global rob_id
class MemRsIssue(val b: GlobalConfig) extends Bundle {
  val cmd    = new MemDecodeCmd(b)
  // Global ROB ID
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Mem domain completion interface
class MemRsComplete(val b: GlobalConfig) extends Bundle {
  val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

// Mem domain issue interface combination (Load + Store)
class MemIssueInterface(val b: GlobalConfig) extends Bundle {
  val ld = Decoupled(new MemRsIssue(b))
  val st = Decoupled(new MemRsIssue(b))
  val cf = Decoupled(new MemRsIssue(b))
}

// Mem domain completion interface combination (Load + Store)
class MemCommitInterface(val b: GlobalConfig) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete(b)))
  val st = Flipped(Decoupled(new MemRsComplete(b)))
  val cf = Flipped(Decoupled(new MemRsComplete(b)))
}

// Local Mem reservation station - simple FIFO scheduler
@instantiable
class MemReservationStation(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    // Decoded instruction input (with global rob_id)
    val mem_decode_cmd_i = Flipped(new DecoupledIO(new Bundle {
      val cmd    = new MemDecodeCmd(b)
      // Global ROB ID
      val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
    }))

    // Rs -> MemLoader/MemStorer
    val issue_o  = new MemIssueInterface(b)
    val commit_i = new MemCommitInterface(b)

    // Output completion signal (with global rob_id, single channel)
    val complete_o = Decoupled(new MemRsComplete(b))
  })

  // Simple FIFO queue, only for buffering
  val fifo = Module(new Queue(
    new Bundle {
      val cmd    = new MemDecodeCmd(b)
      val rob_id = UInt(log2Up(b.frontend.rob_entries).W)
    },
    entries = 4
  )) // Small buffer is sufficient

// -----------------------------------------------------------------------------
// Inbound - FIFO enqueue
// -----------------------------------------------------------------------------
  fifo.io.enq <> io.mem_decode_cmd_i

// -----------------------------------------------------------------------------
// Outbound - instruction issue (dispatch based on is_load/is_store)
// -----------------------------------------------------------------------------
  val headEntry = fifo.io.deq.bits

  // Load issue
  io.issue_o.ld.valid       := fifo.io.deq.valid && headEntry.cmd.is_load
  io.issue_o.ld.bits.cmd    := headEntry.cmd
  io.issue_o.ld.bits.rob_id := headEntry.rob_id

  // Store issue
  io.issue_o.st.valid       := fifo.io.deq.valid && headEntry.cmd.is_store
  io.issue_o.st.bits.cmd    := headEntry.cmd
  io.issue_o.st.bits.rob_id := headEntry.rob_id

  // Config issue
  io.issue_o.cf.valid       := fifo.io.deq.valid && headEntry.cmd.is_config
  io.issue_o.cf.bits.cmd    := headEntry.cmd
  io.issue_o.cf.bits.rob_id := headEntry.rob_id

  // FIFO deq.ready - can only dequeue when target unit is ready
  fifo.io.deq.ready :=
    (headEntry.cmd.is_load && io.issue_o.ld.ready) ||
      (headEntry.cmd.is_store && io.issue_o.st.ready) ||
      (headEntry.cmd.is_config && io.issue_o.cf.ready)

// -----------------------------------------------------------------------------
// Completion signal processing - directly forward to global RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.frontend.rob_entries).W), 3))

  completeArb.io.in(0).valid := io.commit_i.ld.valid
  completeArb.io.in(0).bits  := io.commit_i.ld.bits.rob_id
  io.commit_i.ld.ready       := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.commit_i.st.valid
  completeArb.io.in(1).bits  := io.commit_i.st.bits.rob_id
  io.commit_i.st.ready       := completeArb.io.in(1).ready

  completeArb.io.in(2).valid := io.commit_i.cf.valid
  completeArb.io.in(2).bits  := io.commit_i.cf.bits.rob_id
  io.commit_i.cf.ready       := completeArb.io.in(2).ready

  // Forward completion signal (with global rob_id)
  io.complete_o.valid       := completeArb.io.out.valid
  io.complete_o.bits.rob_id := completeArb.io.out.bits
  completeArb.io.out.ready  := io.complete_o.ready
}
