package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.blink.BallRegist
import framework.bbus.pmc.BallCyclePMC
import framework.bbus.cmdrouter.CmdRouter
import framework.bbus.memrouter.MemRouter

/**
 * BBus - Ball总线，管理多个Ball设备的连接和仲裁
 */
class BBus(ballGenerators: Seq[() => BallRegist with Module])
  (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val numBalls = ballGenerators.length

  val io = IO(new Bundle {
    val cmdReq = Vec(numBalls, Flipped(Decoupled(new BallRsIssue)))
    val cmdResp = Vec(numBalls, Decoupled(new BallRsComplete))

    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
    val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
  })

  // 实例化所有注册的Ball
  val balls = ballGenerators.map(gen => Module(gen()))


// -----------------------------------------------------------------------------
// cmd router
// -----------------------------------------------------------------------------
  val cmdRouter = Module(new CmdRouter(numBalls))
  val idle_ball = Wire(Vec(numBalls, Bool()))
  for (i <- 0 until numBalls) {
    idle_ball(i) := balls(i).Blink.cmdReq.ready
  }

  cmdRouter.io.cmdReq_i <> io.cmdReq
  cmdRouter.io.ballIdle := idle_ball

  for (i <- 0 until numBalls) {
    balls(i).Blink.cmdReq.valid := cmdRouter.io.cmdReq_o.valid && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).Blink.cmdReq.bits := cmdRouter.io.cmdReq_o.bits
  }

  cmdRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i =>
    balls(i).Blink.cmdReq.ready && (cmdRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

  for (i <- 0 until numBalls) {
    cmdRouter.io.cmdResp_i(i) <> balls(i).Blink.cmdResp
  }

  io.cmdResp <> cmdRouter.io.cmdResp_o

// -----------------------------------------------------------------------------
// bus router
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// memory router
// -----------------------------------------------------------------------------
  val memoryrouter = Module(new MemRouter(numBalls)(b, p))
  io.sramRead <> memoryrouter.io.sramRead_o
  io.sramWrite <> memoryrouter.io.sramWrite_o
  io.accRead <> memoryrouter.io.accRead_o
  io.accWrite <> memoryrouter.io.accWrite_o

  for(i <- 0 until numBalls){
    memoryrouter.io.sramRead_i(i) <> balls(i).Blink.sramRead
    memoryrouter.io.sramWrite_i(i) <> balls(i).Blink.sramWrite
    memoryrouter.io.accRead_i(i) <> balls(i).Blink.accRead
    memoryrouter.io.accWrite_i(i) <> balls(i).Blink.accWrite
  }

// -----------------------------------------------------------------------------
// PMC - Performance Monitor Counter
// -----------------------------------------------------------------------------
val pmc = Module(new BallCyclePMC(numBalls))

for (i <- 0 until numBalls) {
  pmc.io.cmdReq_i(i).valid := cmdRouter.io.cmdReq_i(i).valid
  pmc.io.cmdReq_i(i).bits := cmdRouter.io.cmdReq_i(i).bits
  pmc.io.cmdResp_o(i).valid := cmdRouter.io.cmdResp_o(i).valid
  pmc.io.cmdResp_o(i).bits := cmdRouter.io.cmdResp_o(i).bits
}

  override lazy val desiredName = "BBus"
}
