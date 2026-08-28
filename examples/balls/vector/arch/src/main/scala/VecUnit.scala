package examples.balls.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import examples.balls.vector.configs.VectorBallParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class VecUnit(val b: GlobalConfig) extends Module {
  val cfg  = VectorBallParam(b)
  val lane = cfg.lane
  val inW  = cfg.inputWidth
  val accW = cfg.outputWidth

  require(lane == 16, "VecUnit vecmat16 requires lane=16")
  require(inW == 8, "VecUnit vecmat16 requires int8 inputs")
  require(accW == 32, "VecUnit vecmat16 requires int32 accumulators")

  val map = b.ballDomain.ballIdMappings
    .find(_.ballName == "VecBall")
    .getOrElse(
      throw new IllegalArgumentException("VecBall not found in config")
    )

  val inBW  = map.inBW
  val outBW = map.outBW
  require(inBW >= 2, "VecUnit vecmat16 requires two read ports")
  require(outBW >= 4, "VecUnit vecmat16 requires four write ports")

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val core = Module(new VecMat16(b, lane, inBW, outBW, inW, accW))
  core.io.cmdReq <> io.cmdReq
  core.io.cmdResp <> io.cmdResp
  for (i <- 0 until inBW) { core.io.bankRead(i) <> io.bankRead(i) }
  for (i <- 0 until outBW) { core.io.bankWrite(i) <> io.bankWrite(i) }
  io.status <> core.io.status
}
