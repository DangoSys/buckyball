package framework.memdomain.frontend.cmd_channel.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.MemDomainParam
import framework.memdomain.frontend.cmd_channel.decoder.MemDecodeCmd
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

// Mem domain issue interface - includes global rob_id
class MemRsIssue(parameter: MemDomainParam)(implicit p: Parameters) extends Bundle {
  val cmd    = new MemDecodeCmd(parameter)
  // Global ROB ID
  val rob_id = UInt(log2Up(parameter.rob_entries).W)
}

// Mem domain completion interface
class MemRsComplete(parameter: MemDomainParam)(implicit p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(parameter.rob_entries).W)
}

// Mem domain issue interface combination (Load + Store)
class MemIssueInterface(parameter: MemDomainParam)(implicit p: Parameters) extends Bundle {
  val ld = Decoupled(new MemRsIssue(parameter))
  val st = Decoupled(new MemRsIssue(parameter))
}

// Mem domain completion interface combination (Load + Store)
class MemCommitInterface(parameter: MemDomainParam)(implicit p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete(parameter)))
  val st = Flipped(Decoupled(new MemRsComplete(parameter)))
}

// Local Mem reservation station - simple FIFO scheduler
@instantiable
class MemReservationStation(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {

    // Decoded instruction input (with global rob_id)
    val mem_decode_cmd_i = Flipped(new DecoupledIO(new Bundle {
      val cmd    = new MemDecodeCmd(parameter)
      // Global ROB ID
      val rob_id = UInt(log2Up(parameter.rob_entries).W)
    }))

    // Rs -> MemLoader/MemStorer
    val issue_o  = new MemIssueInterface(parameter)
    val commit_i = new MemCommitInterface(parameter)

    // Output completion signal (with global rob_id, single channel)
    val complete_o = Decoupled(new MemRsComplete(parameter))
  })

  // Simple FIFO queue, only for buffering
  val fifo = Module(new Queue(
    new Bundle {
      val cmd    = new MemDecodeCmd(parameter)
      val rob_id = UInt(log2Up(parameter.rob_entries).W)
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

  // FIFO deq.ready - can only dequeue when target unit is ready
  fifo.io.deq.ready :=
    (headEntry.cmd.is_load && io.issue_o.ld.ready) ||
      (headEntry.cmd.is_store && io.issue_o.st.ready)

// -----------------------------------------------------------------------------
// Completion signal processing - directly forward to global RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(parameter.rob_entries).W), 2))

  completeArb.io.in(0).valid := io.commit_i.ld.valid
  completeArb.io.in(0).bits  := io.commit_i.ld.bits.rob_id
  io.commit_i.ld.ready       := completeArb.io.in(0).ready

  completeArb.io.in(1).valid := io.commit_i.st.valid
  completeArb.io.in(1).bits  := io.commit_i.st.bits.rob_id
  io.commit_i.st.ready       := completeArb.io.in(1).ready

  // Forward completion signal (with global rob_id)
  io.complete_o.valid       := completeArb.io.out.valid
  io.complete_o.bits.rob_id := completeArb.io.out.bits
  completeArb.io.out.ready  := io.complete_o.ready
}
