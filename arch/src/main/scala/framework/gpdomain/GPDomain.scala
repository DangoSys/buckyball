package framework.gpdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.gpdomain.sequencer.decoder.DomainDecoder
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

object GpDomainParam {
  implicit def rw: upickle.default.ReadWriter[GpDomainParam] = upickle.default.macroRW

  /**
   * Load from JSON file
   */
  def fromJson(path: String): GpDomainParam = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    upickle.default.read[GpDomainParam](jsonStr)
  }

  /**
   * Generate from global config
   */
  def fromGlobal(global: framework.builtin.BaseConfig): GpDomainParam = {
    GpDomainParam(
      rob_entries = global.rob_entries
    )
  }

}

case class GpDomainParam(
  rob_entries: Int)
    extends SerializableModuleParameter {
  override def toString: String =
    s"""GpDomainParam
       |  ROB entries: $rob_entries
       |""".stripMargin
}

/**
 * General Purpose Domain
 */
@instantiable
class GpDomain(val parameter: GpDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[GpDomainParam] {

  @public
  val io = IO(new Bundle {
    // Receive instructions from GlobalRS
    val global_issue_i = Flipped(Decoupled(new GlobalRsIssue(parameter.rob_entries)))

    // Report completion to GlobalRS
    val global_complete_o = Decoupled(new GlobalRsComplete(parameter.rob_entries))

    // Status signal
    val busy = Output(Bool())
  })

  io.global_issue_i.ready := io.global_complete_o.ready

// -----------------------------------------------------------------------------
// Decode Stage
// -----------------------------------------------------------------------------
  val decoder: Instance[framework.gpdomain.sequencer.decoder.DomainDecoder] =
    Instantiate(new framework.gpdomain.sequencer.decoder.DomainDecoder(parameter))
  decoder.io.inst_i := io.global_issue_i.bits.cmd.raw_cmd
  val decoded = decoder.io.decoded_o

  io.global_complete_o.valid       := io.global_issue_i.valid
  io.global_complete_o.bits.rob_id := io.global_issue_i.bits.rob_id

  io.busy := false.B

  override lazy val desiredName = "GpDomain"
}
