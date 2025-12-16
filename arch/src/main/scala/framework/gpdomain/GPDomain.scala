package framework.gpdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.globalrs.{GlobalRsIssue, GlobalRsComplete}
import framework.gpdomain.sequencer.decoder.DomainDecoder

/**
 * General Purpose Domain
 */
class GpDomainIO(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  // Receive instructions from GlobalRS
  val global_issue_i = Flipped(Decoupled(new GlobalRsIssue))

  // Report completion to GlobalRS
  val global_complete_o = Decoupled(new GlobalRsComplete)

  // Status signal
  val busy = Output(Bool())
}

class GpDomain(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new GpDomainIO)

  io.global_issue_i.ready := io.global_complete_o.ready

// -----------------------------------------------------------------------------
// Decode Stage
// -----------------------------------------------------------------------------
  val decoder = Module(new DomainDecoder)
  decoder.io.inst_i := io.global_issue_i.bits.cmd.raw_cmd
  val decoded = decoder.io.decoded_o

  io.global_complete_o.valid := io.global_issue_i.valid
  io.global_complete_o.bits.rob_id := io.global_issue_i.bits.rob_id

  io.busy := false.B

  override lazy val desiredName = "GpDomain"
}
