package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import framework.top.GlobalConfig
import examples.toy.balldomain.bbus.BBusModule
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite, SubRobRow}
import framework.balldomain.rs.BallReservationStation
import framework.top.channels.{ChannelClusterIO, ChannelIO}

@instantiable
class BallDomain(val b: GlobalConfig) extends Module {
  val memChannel     = b.top.ballMemChannelNum
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val global_issue_i = IO(Flipped(Decoupled(new GlobalRsIssue(b))))

  @public
  val global_complete_o = IO(Decoupled(new GlobalRsComplete(b)))

  @public
  val bankRead = IO(Vec(totalBallRead, Flipped(new BankRead(b))))

  @public
  val bankWrite = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))

  @public
  val subRobReq = IO(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))

  val bbus:        Instance[BBusModule]             = Instantiate(new BBusModule(b))
  val ballDecoder: Instance[BallDomainDecoder]      = Instantiate(new BallDomainDecoder(b))
  val ballRs:      Instance[BallReservationStation] = Instantiate(new BallReservationStation(b))

//---------------------------------------------------------------------------
// Global RS -> Decoder (receive global issue and construct PostGDCmd)
//---------------------------------------------------------------------------

  // Convert global RS issue to Decoder input format
  ballDecoder.cmd_i.valid := global_issue_i.valid
  ballDecoder.cmd_i.bits  := global_issue_i.bits.cmd
  global_issue_i.ready    := ballDecoder.cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> Local BallRS (multi-channel issue to each Ball device)
//---------------------------------------------------------------------------

  // Connect decoded instruction and global rob_id
  ballRs.ball_decode_cmd_i.valid           := ballDecoder.ball_decode_cmd_o.valid
  ballRs.ball_decode_cmd_i.bits.cmd        := ballDecoder.ball_decode_cmd_o.bits
  ballRs.ball_decode_cmd_i.bits.rob_id     := global_issue_i.bits.rob_id
  ballRs.ball_decode_cmd_i.bits.is_sub     := global_issue_i.bits.is_sub
  ballRs.ball_decode_cmd_i.bits.sub_rob_id := global_issue_i.bits.sub_rob_id
  ballDecoder.ball_decode_cmd_o.ready      := ballRs.ball_decode_cmd_i.ready

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

  // BBus -> SubROB
  for (i <- 0 until b.ballDomain.ballNum) {
    subRobReq(i) <> bbus.subRobReq(i)
  }

//---------------------------------------------------------------------------
// Local RS completion signal -> Global RS (single channel, includes global rob_id)
//---------------------------------------------------------------------------
  global_complete_o.valid           := ballRs.complete_o.valid
  global_complete_o.bits.rob_id     := ballRs.complete_o.bits.rob_id
  global_complete_o.bits.is_sub     := ballRs.complete_o.bits.is_sub
  global_complete_o.bits.sub_rob_id := ballRs.complete_o.bits.sub_rob_id
  ballRs.complete_o.ready           := global_complete_o.ready

}
