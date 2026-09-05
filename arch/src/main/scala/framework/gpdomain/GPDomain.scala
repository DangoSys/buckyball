package framework.gpdomain

import chisel3._
import chisel3.util._
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

@instantiable
class GpDomain(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val global_issue_i    = Flipped(Decoupled(new GlobalSchedIssue(b)))
    val global_complete_o = Decoupled(new GlobalSchedComplete(b))
    // Status signal
    val busy              = Output(Bool())
  })

// -----------------------------------------------------------------------------
// Decode Stage
// -----------------------------------------------------------------------------
  val decoder: Instance[framework.gpdomain.sequencer.decoder.DomainDecoder] =
    Instantiate(new framework.gpdomain.sequencer.decoder.DomainDecoder(b))
  // Extract raw_inst from PostGDCmd
  decoder.io.inst_i <> io.global_issue_i.bits.cmd.cmd
  val decoded = decoder.io.decoded_o

  // GP operations are architecturally fire-and-forget, but completing them
  // combinationally feeds the scheduler's completion arbiter in the same
  // cycle. A one-entry queue preserves II=1 while cutting that issue ->
  // complete path at a register boundary.
  val completeQ = Module(new Queue(new GlobalSchedComplete(b), 1, pipe = true))
  io.global_issue_i.ready          := completeQ.io.enq.ready
  completeQ.io.enq.valid           := io.global_issue_i.valid
  completeQ.io.enq.bits.rob_id     := io.global_issue_i.bits.rob_id
  completeQ.io.enq.bits.is_sub     := io.global_issue_i.bits.is_sub
  completeQ.io.enq.bits.sub_rob_id := io.global_issue_i.bits.sub_rob_id
  io.global_complete_o <> completeQ.io.deq

  io.busy := false.B

}
