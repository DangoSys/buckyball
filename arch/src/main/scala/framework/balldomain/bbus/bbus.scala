package framework.balldomain.bbus

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.toy.balldomain.BallDomainParam
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
class BBus(val parameter: BallDomainParam, ballGenerators: Seq[() => BallRegist with Module]) extends Module {
  val numBalls = ballGenerators.length

  @public
  val cmdReq = IO(Vec(numBalls, Flipped(Decoupled(new BallRsIssue(parameter)))))

  @public
  val cmdResp = IO(Vec(numBalls, Decoupled(new BallRsComplete(parameter))))

  @public
  val bankRead = IO(Vec(
    parameter.numBanks,
    Flipped(new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.numBanks))
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

  // Instantiate all registered Balls
  val balls = ballGenerators.map(gen => Module(gen()))

// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------
  val cmdRouter = Module(new CmdRouter(parameter, numBalls))
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
  val memoryrouter = Module(new MemRouter(parameter, numBalls, parameter.bbusChannel))
  memoryrouter.io.bbusConfig_i <> cmdRouter.io.bbusConfig_o

  // Direct connection from balls to memory router (no ToVirtualLine)
  for (i <- 0 until numBalls) {
    for (j <- 0 until parameter.numBanks) {
      memoryrouter.io.bankRead_i(i)(j) <> balls(i).Blink.bankRead(j)
      memoryrouter.io.bankWrite_i(i)(j) <> balls(i).Blink.bankWrite(j)
    }
  }

  bankRead <> memoryrouter.io.bankRead_o
  bankWrite <> memoryrouter.io.bankWrite_o

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
  val pmc = Module(new BallCyclePMC(parameter, numBalls))

  for (i <- 0 until numBalls) {
    pmc.io.cmdReq_i(i).valid  := cmdRouter.io.cmdReq_i(i).fire
    pmc.io.cmdReq_i(i).bits   := cmdRouter.io.cmdReq_i(i).bits
    // Remove delay caused by RoB blocking preventing commit
    pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
    pmc.io.cmdResp_o(i).bits  := cmdRouter.io.cmdResp_o(i).bits
  }

  override lazy val desiredName = "BBus"
}
