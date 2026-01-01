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
  for (i <- 0 until numBalls) {
    for (j <- 0 until bbusChannel) {
      memoryrouter.io.bankRead_i(i)(j) <> balls(i).Blink.bankRead(j)
      memoryrouter.io.bankWrite_i(i)(j) <> balls(i).Blink.bankWrite(j)
    }
  }

  bankRead <> memoryrouter.io.bankRead_o
  bankWrite <> memoryrouter.io.bankWrite_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    // Remove delay caused by RoB blocking preventing commit
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

}
