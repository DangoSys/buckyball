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

class BBusConfigIO(numBalls: Int) extends Bundle {
  val src_bid = UInt(log2Ceil(numBalls).W)
  val dst_bid = UInt(log2Ceil(numBalls).W)
  val set     = Bool()
}

/**
 * BBus - Ball bus, manages connections and arbitration of multiple Ball devices
 */
@instantiable
class BBus(val b: GlobalConfig, ballGenerators: Seq[() => BallRegist with Module]) extends Module {
  val numBalls = ballGenerators.length

  @public
  val cmdReq = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(b)))))

  @public
  val cmdResp = IO(Vec(numBalls, Decoupled(new BallRsComplete(b))))

  @public
  val bankRead = IO(Vec(
    b.memDomain.bankNum,
    Flipped(new BankRead(b))
  ))

  @public
  val bankWrite = IO(Vec(
    b.memDomain.bankNum,
    Flipped(new BankWrite(b))
  ))

  // Instantiate all registered Balls
  // Note: Since Instantiate requires 'new' expression and ballGenerators are functions,
  // we use Module here. The balls themselves are @instantiable, but when instantiated
  // from a function generator pattern, we use Module. This is acceptable as the balls
  // are still properly instantiated and can be used with the hierarchy system.
  val balls = ballGenerators.map(gen => gen())

// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------
  val cmdRouter: Instance[CmdRouter] = Instantiate(new CmdRouter(b, numBalls))
  val idle_ball = Wire(Vec(numBalls, Bool()))
  for (i <- 0 until numBalls) {
    idle_ball(i) := balls(i).Blink.cmdReq.ready
  }

  cmdRouter.io.cmdReq_i <> cmdReq
  cmdRouter.io.ballIdle := idle_ball

  for (i <- 0 until numBalls) {
    balls(i).Blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).Blink.cmdReq.bits  := cmdRouter.io.cmdReq_o.bits
  }

  cmdRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i =>
    balls(i).Blink.cmdReq.ready && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

  for (i <- 0 until numBalls) {
    cmdRouter.io.cmdResp_i(i) <> balls(i).Blink.cmdResp
  }

  cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// memory router
// -----------------------------------------------------------------------------
  val memoryrouter: Instance[MemRouter] = Instantiate(new MemRouter(b, numBalls, b.memDomain.bankChannel))
  memoryrouter.io.bbusConfig_i <> cmdRouter.io.bbusConfig_o

  // Direct connection from balls to memory router (no ToVirtualLine)
  for (i <- 0 until numBalls) {
    for (j <- 0 until b.memDomain.bankNum) {
      memoryrouter.io.bankRead_i(i)(j) <> balls(i).Blink.bankRead(j)
      memoryrouter.io.bankWrite_i(i)(j) <> balls(i).Blink.bankWrite(j)
    }
  }

  bankRead <> memoryrouter.io.bankRead_o
  bankWrite <> memoryrouter.io.bankWrite_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  val pmc = Module(new BallCyclePMC(b, numBalls))

  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    // Remove delay caused by RoB blocking preventing commit
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

}
