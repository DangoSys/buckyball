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

  io.global_issue_i.ready := io.global_complete_o.ready

// -----------------------------------------------------------------------------
// Decode Stage
// -----------------------------------------------------------------------------
  val decoder: Instance[framework.gpdomain.sequencer.decoder.DomainDecoder] =
    Instantiate(new framework.gpdomain.sequencer.decoder.DomainDecoder(b))
  // Extract raw_inst from PostGDCmd
  decoder.io.inst_i <> io.global_issue_i.bits.cmd.cmd
  val decoded = decoder.io.decoded_o

  io.global_complete_o.valid           := io.global_issue_i.valid
  io.global_complete_o.bits.rob_id     := io.global_issue_i.bits.rob_id
  io.global_complete_o.bits.is_sub     := io.global_issue_i.bits.is_sub
  io.global_complete_o.bits.sub_rob_id := io.global_issue_i.bits.sub_rob_id

  io.busy := false.B

}
