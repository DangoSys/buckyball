package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.HasBlink
import framework.balldomain.bbus.pmc.BallCyclePMC
import framework.balldomain.bbus.cmdrouter.CmdRouter
import framework.balldomain.bbus.memrouter.MemRouter
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.top.channels.{Channel, ChannelClusterIO, ChannelIO}
import framework.balldomain.bbus.memrouter.{FreeChannelResp, PeakChannelReq}

/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices
 */
@instantiable
class BBus(val b: GlobalConfig, ballGenerators: Seq[() => HasBlink with Module]) extends Module {
  val numBalls          = b.ballDomain.ballNum
  val ballMemChannelNum = b.top.ballMemChannelNum
  val memBallChannelNum = b.top.memBallChannelNum
  val totalBallRead     = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite    = b.ballDomain.ballIdMappings.map(_.outBW).sum

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

  // bbus - mem
  // Channel interface using ChannelClusterIO
  // For ballMemChannel: bbus outputs to channel.in, receives from channel.out
  // So we need Flipped because direction is reversed from ChannelClusterIO's perspective
  @public
  val ballMemChannel = IO(Flipped(new ChannelClusterIO(b, ballMemChannelNum)))

  @public
  val memBallChannel = IO(new ChannelClusterIO(b, memBallChannelNum))

  val balls = ballGenerators.map(gen => Module(gen()))
  val cmdRouter:    Instance[CmdRouter]    = Instantiate(new CmdRouter(b))
  val memoryrouter: Instance[MemRouter]    = Instantiate(new MemRouter(b))
  val pmc:          Instance[BallCyclePMC] = Instantiate(new BallCyclePMC(b))

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
// memory router
// -----------------------------------------------------------------------------
  memoryrouter.io.bankRead_i <> bankRead
  memoryrouter.io.bankWrite_i <> bankWrite
  memoryrouter.io.peakChannelReq <> ballMemChannel.peakChannelReq
  memoryrouter.io.freeChannelResp <> ballMemChannel.freeChannelResp

  // Connect channel outputs (from channel.out in top level) to memory router
  // Channels are unified, router handles routing to bankRead_o or bankWrite_o based on request type
  // For now, we route to both and let router decide based on internal logic
  // TODO: Router should determine read/write based on request metadata
  for (i <- 0 until ballMemChannelNum) {
    // Connect to both read and write outputs, router will select based on request type
    memoryrouter.io.bankRead_o(i).io.req.valid      := ballMemChannel.channelOut(i).data.valid
    memoryrouter.io.bankRead_o(i).io.req.bits.addr  := ballMemChannel.channelOut(i).data.bits
    memoryrouter.io.bankWrite_o(i).io.req.valid     := ballMemChannel.channelOut(i).data.valid
    memoryrouter.io.bankWrite_o(i).io.req.bits.addr := ballMemChannel.channelOut(i).data.bits
    ballMemChannel.channelOut(i).data.ready         := memoryrouter.io.bankRead_o(
      i
    ).io.req.ready || memoryrouter.io.bankWrite_o(i).io.req.ready
  }

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

}
