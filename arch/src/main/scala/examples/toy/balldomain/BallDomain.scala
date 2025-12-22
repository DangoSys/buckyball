package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.toy.balldomain.rs.BallRSModule
import examples.toy.balldomain.bbus.BBusModule
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite}

object BallDomainParam {
  implicit def rw: upickle.default.ReadWriter[BallDomainParam] = upickle.default.macroRW

  /**
   * Load from JSON file
   */
  def fromJson(path: String): BallDomainParam = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    upickle.default.read[BallDomainParam](jsonStr)
  }

  /**
   * Generate from global config
   */
  def fromGlobal(global: framework.builtin.BaseConfig): BallDomainParam = {
    BallDomainParam(
      rob_entries = global.rob_entries,
      numBanks = global.bankNum,
      bbusChannel = global.bankChannel,
      bankEntries = global.bankEntries,
      bankWidth = global.bankWidth,
      bankMaskLen = global.bankMaskLen,
      emptyBallid = global.emptyBallid
    )
  }

}

case class BallDomainParam(
  rob_entries: Int,
  numBanks:    Int,
  bbusChannel: Int,
  bankEntries: Int,
  bankWidth:   Int,
  bankMaskLen: Int,
  emptyBallid: Int = 5)
    extends SerializableModuleParameter {
  override def toString: String =
    s"""BallDomainParam
       |  ROB entries: $rob_entries
       |  Num banks: $numBanks
       |  BBus channel: $bbusChannel
       |  Bank entries: $bankEntries
       |  Bank width: $bankWidth
       |  Bank mask length: $bankMaskLen
       |  Empty Ball ID: $emptyBallid
       |""".stripMargin
}

/**
 * Ball Domain top level - uses new simplified BBus architecture
 */
@instantiable
class BallDomain(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {

  @public
  val global_issue_i = IO(Flipped(Decoupled(new GlobalRsIssue(parameter.rob_entries))))

  @public
  val global_complete_o = IO(Decoupled(new GlobalRsComplete(parameter.rob_entries)))

  @public
  val bankRead = IO(Vec(
    parameter.numBanks,
    Flipped(new BankRead(
      parameter.bankEntries,
      parameter.bankWidth,
      parameter.rob_entries,
      parameter.numBanks
    ))
  ))

  @public
  val bankWrite = IO(Vec(
    parameter.numBanks,
    Flipped(new BankWrite(
      parameter.bankEntries,
      parameter.bankWidth,
      parameter.bankMaskLen,
      parameter.rob_entries,
      parameter.numBanks
    ))
  ))

  // Create new BBus module
  val bbus: Instance[BBusModule] = Instantiate(new BBusModule(parameter))

//---------------------------------------------------------------------------
// Global RS -> Decoder (receive global issue and construct PostGDCmd)
//---------------------------------------------------------------------------
  val ballDecoder: Instance[BallDomainDecoder] = Instantiate(new BallDomainDecoder(parameter)(p))

  // Convert global RS issue to Decoder input format
  ballDecoder.raw_cmd_i.valid := global_issue_i.valid
  ballDecoder.raw_cmd_i.bits  := global_issue_i.bits.cmd
  global_issue_i.ready        := ballDecoder.raw_cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> Local BallRS (multi-channel issue to each Ball device)
//---------------------------------------------------------------------------
  val ballRs: Instance[BallRSModule] = Instantiate(new BallRSModule(parameter))

  // Connect decoded instruction and global rob_id
  ballRs.ball_decode_cmd_i.valid       := ballDecoder.ball_decode_cmd_o.valid
  ballRs.ball_decode_cmd_i.bits.cmd    := ballDecoder.ball_decode_cmd_o.bits
  ballRs.ball_decode_cmd_i.bits.rob_id := global_issue_i.bits.rob_id
  ballDecoder.ball_decode_cmd_o.ready  := ballRs.ball_decode_cmd_i.ready

//---------------------------------------------------------------------------
// Local BallRS -> BBus (multi-channel)
//---------------------------------------------------------------------------
  bbus.cmdReq <> ballRs.issue_o.balls
  ballRs.commit_i.balls <> bbus.cmdResp

//---------------------------------------------------------------------------
// BBus -> Mem Domain
//---------------------------------------------------------------------------
  bbus.bankRead <> bankRead
  bbus.bankWrite <> bankWrite

//---------------------------------------------------------------------------
// Local RS completion signal -> Global RS (single channel, includes global rob_id)
//---------------------------------------------------------------------------
  global_complete_o <> ballRs.complete_o

  override lazy val desiredName = "BallDomain"
}
