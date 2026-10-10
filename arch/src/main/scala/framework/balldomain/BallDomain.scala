package framework.balldomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.bbus.BBus
import framework.balldomain.decoder.BallDomainDecoder
import framework.balldomain.rs.BallReservationStation
import framework.balldomain.blink.{BankRead, BankWrite, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue}
import framework.rvv.RvvBallCommand
import framework.frontend.decoder.{DomainId, PostGDCmd}

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
  val totalMmioRead  = b.ballDomain.ballIdMappings.map(_.mmioReadBW).sum
  val totalMmioWrite = b.ballDomain.ballIdMappings.map(_.mmioWriteBW).sum

  @public
  val global_issue_i           = IO(Flipped(Decoupled(new GlobalSchedIssue(b))))
  @public
  val global_complete_o        = IO(Decoupled(new GlobalSchedComplete(b)))
  @public
  val ballChannelActive        = IO(Output(Vec(b.ballDomain.ballNum, Bool())))
  @public
  val ballChannelReady         = IO(Input(Vec(b.ballDomain.ballNum, Bool())))
  @public
  val bankRead                 = IO(Vec(totalBallRead, Flipped(new BankRead(b))))
  @public
  val bankWrite                = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))
  @public
  val mmioRead                 = IO(Vec(totalMmioRead, Flipped(new MmioRead(b))))
  @public
  val mmioWrite                = IO(Vec(totalMmioWrite, Flipped(new MmioWrite(b))))
  @public
  val subRobReq                = IO(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))
  @public val kernelBanks      = if (b.rvv.enable) Some(IO(Input(new framework.rvv.BankLayout(b)))) else None
  @public val internalCommand  = if (b.rvv.enable) Some(IO(Flipped(Decoupled(new RvvBallCommand)))) else None
  @public val internalComplete = if (b.rvv.enable) Some(IO(Decoupled(UInt(64.W)))) else None
  @public val ownerRobId       = if (b.rvv.enable) Some(IO(Input(UInt(log2Up(b.frontend.rob_entries).W)))) else None

  val bbus:        Instance[BBus]                   = Instantiate(new BBus(b))
  val ballDecoder: Instance[BallDomainDecoder]      = Instantiate(new BallDomainDecoder(b))
  val ballRs:      Instance[BallReservationStation] = Instantiate(new BallReservationStation(b))

  if (b.rvv.enable) {
    bbus.kernelBanks.get := kernelBanks.get
    val decoder = Instantiate(new BallDomainDecoder(b))
    decoder.cmd_i.valid                      := internalCommand.get.valid
    decoder.cmd_i.bits                       := 0.U.asTypeOf(new PostGDCmd(b))
    decoder.cmd_i.bits.domain_id             := DomainId.BALL
    decoder.cmd_i.bits.op1_col               := 1.U
    decoder.cmd_i.bits.op2_col               := 1.U
    decoder.cmd_i.bits.wr_col                := 1.U
    decoder.cmd_i.bits.cmd.funct             := internalCommand.get.bits.funct7
    decoder.cmd_i.bits.cmd.rs1Data           := internalCommand.get.bits.rs1
    decoder.cmd_i.bits.cmd.rs2Data           := internalCommand.get.bits.rs2
    internalCommand.get.ready                := decoder.cmd_i.ready
    bbus.internalCommand.get.valid           := decoder.ball_decode_cmd_o.valid
    bbus.internalCommand.get.bits.cmd        := decoder.ball_decode_cmd_o.bits
    bbus.internalCommand.get.bits.rob_id     := ownerRobId.get
    bbus.internalCommand.get.bits.is_sub     := false.B
    bbus.internalCommand.get.bits.sub_rob_id := 0.U
    decoder.ball_decode_cmd_o.ready          := bbus.internalCommand.get.ready
    internalComplete.get <> bbus.internalComplete.get
  }

//===-------------------------------------------------------------------===//
// Global RS -> Decoder
//===-------------------------------------------------------------------===//
  ballDecoder.cmd_i.valid := global_issue_i.valid
  ballDecoder.cmd_i.bits  := global_issue_i.bits.cmd
  global_issue_i.ready    := ballDecoder.cmd_i.ready

//===-------------------------------------------------------------------===//
// Decoder -> Local BallRS
//===-------------------------------------------------------------------===//
  ballRs.ball_decode_cmd_i.valid           := ballDecoder.ball_decode_cmd_o.valid
  ballRs.ball_decode_cmd_i.bits.cmd        := ballDecoder.ball_decode_cmd_o.bits
  ballRs.ball_decode_cmd_i.bits.rob_id     := global_issue_i.bits.rob_id
  ballRs.ball_decode_cmd_i.bits.is_sub     := global_issue_i.bits.is_sub
  ballRs.ball_decode_cmd_i.bits.sub_rob_id := global_issue_i.bits.sub_rob_id
  ballDecoder.ball_decode_cmd_o.ready      := ballRs.ball_decode_cmd_i.ready

//===-------------------------------------------------------------------===//
// Local BallRS -> BBus
//===-------------------------------------------------------------------===//
  bbus.cmdReq <> ballRs.issue_o.balls
  ballRs.commit_i.balls <> bbus.cmdResp

//===-------------------------------------------------------------------===//
// BBus -> Mem Domain
//===-------------------------------------------------------------------===//
  bbus.bankRead <> bankRead
  bbus.bankWrite <> bankWrite
  ballChannelActive     := bbus.ballChannelActive
  bbus.ballChannelReady := ballChannelReady
  bbus.mmioRead <> mmioRead
  bbus.mmioWrite <> mmioWrite

  for (i <- 0 until b.ballDomain.ballNum) {
    subRobReq(i) <> bbus.subRobReq(i)
  }

//===-------------------------------------------------------------------===//
// Local RS completion -> Global RS
//===-------------------------------------------------------------------===//
  global_complete_o.valid           := ballRs.complete_o.valid
  global_complete_o.bits.rob_id     := ballRs.complete_o.bits.rob_id
  global_complete_o.bits.is_sub     := ballRs.complete_o.bits.is_sub
  global_complete_o.bits.sub_rob_id := ballRs.complete_o.bits.sub_rob_id
  global_complete_o.bits.fault      := 0.U.asTypeOf(global_complete_o.bits.fault)
  ballRs.complete_o.ready           := global_complete_o.ready
}
