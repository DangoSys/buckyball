package examples.toy.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import framework.top.GlobalConfig
import examples.toy.balldomain.bbus.BBusModule
import framework.frontend.globalrs.{GlobalRsComplete, GlobalRsIssue}
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.balldomain.rs.BallReservationStation
import framework.top.channels.ChannelIO
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

@instantiable
class BallDomain(val b: GlobalConfig) extends Module {
  val memChannel = b.top.ballMemChannelProducer

  @public
  val global_issue_i = IO(Flipped(Decoupled(new GlobalRsIssue(b))))

  @public
  val global_complete_o = IO(Decoupled(new GlobalRsComplete(b)))

  @public
  val bankRead = IO(Vec(memChannel, Flipped(new BankRead(b))))

  @public
  val bankWrite = IO(Vec(memChannel, Flipped(new BankWrite(b))))

  @public
  val ballMemChannel = IO(new Bundle {
    // Output to channel.in (channel.in is Flipped, so this is ChannelIO)
    val channelIn       = Vec(memChannel, new ChannelIO(b))
    // Input from channel.out (channel.out is ChannelIO, so this is Flipped)
    val channelOut      = Vec(memChannel, Flipped(new ChannelIO(b)))
    val peakChannelReq  = Flipped(Decoupled(new PeakChannelReq(b)))
    val freeChannelResp = Decoupled(new FreeChannelResp(b))
  })

  @public
  val memBallChannelIn  = IO(Vec(b.top.ballMemChannelConsumer, Flipped(new ChannelIO(b))))
  @public
  val memBallChannelOut = IO(Vec(b.top.ballMemChannelConsumer, new ChannelIO(b)))

  val bbus:        Instance[BBusModule]             = Instantiate(new BBusModule(b))
  val ballDecoder: Instance[BallDomainDecoder]      = Instantiate(new BallDomainDecoder(b))
  val ballRs:      Instance[BallReservationStation] = Instantiate(new BallReservationStation(b))

//---------------------------------------------------------------------------
// Global RS -> Decoder (receive global issue and construct PostGDCmd)
//---------------------------------------------------------------------------

  // Convert global RS issue to Decoder input format
  ballDecoder.raw_cmd_i.valid := global_issue_i.valid
  ballDecoder.raw_cmd_i.bits  := global_issue_i.bits.cmd
  global_issue_i.ready        := ballDecoder.raw_cmd_i.ready

//---------------------------------------------------------------------------
// Decoder -> Local BallRS (multi-channel issue to each Ball device)
//---------------------------------------------------------------------------

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
  bbus.ballMemChannel <> ballMemChannel
  bbus.memBallChannelIn <> memBallChannelIn
  bbus.memBallChannelOut <> memBallChannelOut

//---------------------------------------------------------------------------
// Local RS completion signal -> Global RS (single channel, includes global rob_id)
//---------------------------------------------------------------------------
  global_complete_o <> ballRs.complete_o

}
