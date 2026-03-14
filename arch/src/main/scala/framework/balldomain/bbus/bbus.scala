package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.HasBlink
import framework.balldomain.bbus.pmc.BallCyclePMC
import framework.balldomain.bbus.cmdrouter.CmdRouter
import framework.balldomain.blink.{BankRead, BankWrite, SubRobRow}
import framework.top.channels.{Channel, ChannelClusterIO, ChannelIO}

/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices
 */
@instantiable
class BBus(val b: GlobalConfig, ballGenerators: Seq[() => HasBlink with Module]) extends Module {
  val numBalls       = b.ballDomain.ballNum
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  // Rs - bbus - balls
  @public
  val cmdReq    = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b)))))
  @public
  val cmdResp   = IO(Vec(numBalls, Decoupled(new BallRsComplete(b))))
  // balls - bbus
  @public
  val bankRead  = IO(Vec(totalBallRead, Flipped(new BankRead(b))))
  @public
  val bankWrite = IO(Vec(totalBallWrite, Flipped(new BankWrite(b))))
  // balls - bbus - SubROB
  @public
  val subRobReq = IO(Vec(numBalls, Decoupled(new SubRobRow(b))))

  val balls = ballGenerators.map(gen => Module(gen()))
  val cmdRouter: Instance[CmdRouter]    = Instantiate(new CmdRouter(b))
  val pmc:       Instance[BallCyclePMC] = Instantiate(new BallCyclePMC(b))

// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------

  val idle_ball = VecInit(balls.map(_.blink.cmdReq.ready))

  cmdRouter.io.cmdReq_i <> cmdReq
  cmdRouter.io.ballIdle := idle_ball

  for (i <- 0 until numBalls) {
    balls(i).blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).blink.cmdReq.bits  := cmdRouter.io.cmdReq_o.bits

    cmdRouter.io.cmdResp_i(i) <> balls(i).blink.cmdResp
  }

  cmdRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i =>
    balls(i).blink.cmdReq.ready && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

  cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

// Connect balls' bankRead and bankWrite to memrouter
  var readChannelIdx  = 0
  var writeChannelIdx = 0

  for (ball <- balls) {
    val ballConfig = b.ballDomain.ballIdMappings.find(_.ballName == ball.getClass.getSimpleName)
    val inBW       = ballConfig.map(_.inBW).getOrElse(0)
    val outBW      = ballConfig.map(_.outBW).getOrElse(0)

    for (i <- 0 until inBW) {
      bankRead(readChannelIdx) <> ball.blink.bankRead(i)
      readChannelIdx = readChannelIdx + 1
    }

    for (i <- 0 until outBW) {
      bankWrite(writeChannelIdx) <> ball.blink.bankWrite(i)
      writeChannelIdx = writeChannelIdx + 1
    }
  }

  // Connect balls' subRobReq
  for (i <- 0 until numBalls) {
    subRobReq(i) <> balls(i).blink.subRobReq
  }

}
