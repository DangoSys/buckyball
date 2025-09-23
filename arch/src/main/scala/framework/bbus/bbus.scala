package framework.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.blink.BallRegist

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
  val cmdReqRouter = Module(new CmdReqRouter(numBalls))
  val idle_ball = Wire(Vec(numBalls, Bool()))
  for (i <- 0 until numBalls) {
    idle_ball(i) := balls(i).Blink.cmdReq.ready
  }

  cmdReqRouter.io.cmdReq_i <> io.cmdReq
  cmdReqRouter.io.ballIdle <> idle_ball

  for (i <- 0 until numBalls) {
    balls(i).Blink.cmdReq.valid := cmdReqRouter.io.cmdReq_o.valid && (cmdReqRouter.io.cmdReq_o.bits.cmd.bid === i.U)
    balls(i).Blink.cmdReq.bits := cmdReqRouter.io.cmdReq_o.bits
  }
  
  // ready信号：只要目标ball准备好接收，就可以ready
  cmdReqRouter.io.cmdReq_o.ready := VecInit((0 until numBalls).map(i => 
    balls(i).Blink.cmdReq.ready && (cmdReqRouter.io.cmdReq_o.bits.cmd.bid === i.U)
  )).asUInt.orR

// -----------------------------------------------------------------------------
// cmd resp
// -----------------------------------------------------------------------------
  // val cmdRespRouter = Module(new CmdRespRouter(numBalls))

  // for (i <- 0 until numBalls) {
  //   cmdRespRouter.io.cmdResp_i(i) <> balls(i).Blink.cmdResp
  // }
  // io.cmdResp <> cmdRespRouter.io.cmdResp_o

  for (i <- 0 until numBalls) {
    io.cmdResp(i) <> balls(i).Blink.cmdResp
  }

// -----------------------------------------------------------------------------
// bus router
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// memory router
// -----------------------------------------------------------------------------
  for (bank <- 0 until b.sp_banks) {
    for (i <- 0 until numBalls) {
      balls(i).Blink.sramRead(bank) <> io.sramRead(bank)
      balls(i).Blink.sramWrite(bank) <> io.sramWrite(bank)
    }
  }

  for (bank <- 0 until b.acc_banks) {
    for (i <- 0 until numBalls) {
      balls(i).Blink.accRead(bank) <> io.accRead(bank)
      balls(i).Blink.accWrite(bank) <> io.accWrite(bank)
    }
  }


  override lazy val desiredName = "BBus"
}
