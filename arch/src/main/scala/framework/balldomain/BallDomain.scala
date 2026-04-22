package framework.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.decoder.BallDomainDecoder
import framework.balldomain.rs.BallReservationStation
import framework.balldomain.blink.{BankRead, BankWrite, SubRobRow}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}

/**
 * Ball domain.
 *
 * BBus reflectively constructs ball instances from `b.ballDomain.ballIdMappings`;
 * `BallDomainDecoder` reads `b.ballDomain.ballISA` to route funct7 → bid.
 * Both are pure data-driven, so no example needs its own subclass.
 */
@instantiable
class BallDomain(val b: GlobalConfig) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val global_issue_i    = IO(Flipped(Decoupled(new GlobalSchedIssue(b))))
  @public
  val global_complete_o = IO(Decoupled(new GlobalSchedComplete(b)))
  @public
  val bankRead          = IO(Vec(totalBallRead, Flipped(new BankRead(b))))
  @public
  val bankWrite         = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))
  @public
  val subRobReq         = IO(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))

  val bbus:        Instance[BBus]                   = Instantiate(new BBus(b))
  val ballDecoder: Instance[BallDomainDecoder]      = Instantiate(new BallDomainDecoder(b))
  val ballRs:      Instance[BallReservationStation] = Instantiate(new BallReservationStation(b))

//---------------------------------------------------------------------------
// Global RS -> Decoder
//---------------------------------------------------------------------------
  ballDecoder.cmd_i.valid := global_issue_i.valid
  ballDecoder.cmd_i.bits  := global_issue_i.bits.cmd
  global_issue_i.ready    := ballDecoder.cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> Local BallRS
//---------------------------------------------------------------------------
  ballRs.ball_decode_cmd_i.valid           := ballDecoder.ball_decode_cmd_o.valid
  ballRs.ball_decode_cmd_i.bits.cmd        := ballDecoder.ball_decode_cmd_o.bits
  ballRs.ball_decode_cmd_i.bits.rob_id     := global_issue_i.bits.rob_id
  ballRs.ball_decode_cmd_i.bits.is_sub     := global_issue_i.bits.is_sub
  ballRs.ball_decode_cmd_i.bits.sub_rob_id := global_issue_i.bits.sub_rob_id
  ballDecoder.ball_decode_cmd_o.ready      := ballRs.ball_decode_cmd_i.ready

//---------------------------------------------------------------------------
// Local BallRS -> BBus
//---------------------------------------------------------------------------
  bbus.cmdReq <> ballRs.issue_o.balls
  ballRs.commit_i.balls <> bbus.cmdResp

//---------------------------------------------------------------------------
// BBus -> Mem Domain
//---------------------------------------------------------------------------
  bbus.bankRead <> bankRead
  bbus.bankWrite <> bankWrite

  for (i <- 0 until b.ballDomain.ballNum) {
    subRobReq(i) <> bbus.subRobReq(i)
  }

//---------------------------------------------------------------------------
// Local RS completion -> Global RS
//---------------------------------------------------------------------------
  global_complete_o.valid           := ballRs.complete_o.valid
  global_complete_o.bits.rob_id     := ballRs.complete_o.bits.rob_id
  global_complete_o.bits.is_sub     := ballRs.complete_o.bits.is_sub
  global_complete_o.bits.sub_rob_id := ballRs.complete_o.bits.sub_rob_id
  ballRs.complete_o.ready           := global_complete_o.ready
}
