package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.BallRegist
import framework.balldomain.bbus.pmc.BallCyclePMC
import framework.balldomain.bbus.cmdrouter.CmdRouter
import framework.balldomain.bbus.memrouter.MemRouter
import framework.balldomain.blink.{BankRead, BankWrite}

/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices
 */
@instantiable
class BBus(val b: GlobalConfig, ballGenerators: Seq[() => BallRegist with Module]) extends Module {
  val numBalls    = b.ballDomain.ballNum
  val bbusChannel = b.ballDomain.bbusChannel

  @public
  val cmdReq    = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b)))))
  @public
  val cmdResp   = IO(Vec(numBalls, Decoupled(new BallRsComplete(b))))
  @public
  val bankRead  = IO(Vec(bbusChannel, Flipped(new BankRead(b))))
  @public
  val bankWrite = IO(Vec(bbusChannel, Flipped(new BankWrite(b))))

  // Instantiate all registered Balls
  val balls = ballGenerators.map(gen => Module(gen()))
  val cmdRouter:    Instance[CmdRouter]    = Instantiate(new CmdRouter(b))
  val memoryrouter: Instance[MemRouter]    = Instantiate(new MemRouter(b))
  val pmc:          Instance[BallCyclePMC] = Instantiate(new BallCyclePMC(b))

// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------

  val idle_ball = Wire(Vec(numBalls, Bool()))
  for (i <- 0 until numBalls) {
    idle_ball(i) := balls(i).Blink.cmdReq.ready
  }

  cmdRouter.io.cmdReq_i <> cmdReq
  cmdRouter.io.ballIdle := idle_ball

  for (i <- 0 until numBalls) {
    balls(i).Blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).Blink.cmdReq.bits  := cmdRouter.io.cmdReq_o.bits

    cmdRouter.io.cmdResp_i(i) <> balls(i).Blink.cmdResp
  }

  cmdRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i =>
    balls(i).Blink.cmdReq.ready && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

  cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// memory router
// -----------------------------------------------------------------------------
  // Connect each ball's bankRead/bankWrite based on its configured bandwidth
  // All requests from balls will go through Router inside MemRouter
  // and become bbusChannel outputs (not bankNum)
  // Build flat index mapping from ball+channel to flat index
  var readFlatIdx  = 0
  var writeFlatIdx = 0
  for (i <- 0 until numBalls) {
    val ballMapping = b.ballDomain.ballIdMappings(i)
    val inBW        = ballMapping.inBW
    val outBW       = ballMapping.outBW

    // Connect all input bandwidth channels (bankRead) to MemRouter
    for (j <- 0 until inBW) {
      memoryrouter.io.bankRead_i(readFlatIdx) <> balls(i).Blink.bankRead(j)
      readFlatIdx += 1
    }

    // Connect all output bandwidth channels (bankWrite) to MemRouter
    for (j <- 0 until outBW) {
      memoryrouter.io.bankWrite_i(writeFlatIdx) <> balls(i).Blink.bankWrite(j)
      writeFlatIdx += 1
    }
  }

  // MemRouter outputs bbusChannel channels (routed from all ball requests)
  bankRead <> memoryrouter.io.bankRead_o
  bankWrite <> memoryrouter.io.bankWrite_o

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
