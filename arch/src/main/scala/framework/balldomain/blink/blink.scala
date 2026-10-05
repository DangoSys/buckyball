package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}

class BlinkPortIO[I <: Data, R <: Data](
  b:        GlobalConfig,
  inBW:     Int,
  outBW:    Int,
  issue:    I,
  complete: R)
    extends Bundle
    with HasBallStatus {
  val status       = new BallStatus()
  val channelReady = Input(Bool())
  val cmdReq       = Flipped(Decoupled(issue))
  val cmdResp      = Decoupled(complete)
  val bankRead     = Vec(inBW, Flipped(new BankRead(b)))
  val bankWrite    = Vec(outBW, Flipped(new BankWrite(b)))
}

class BlinkIO(
  b:           GlobalConfig,
  inBW:        Int,
  outBW:       Int,
  mmioReadBW:  Int = 0,
  mmioWriteBW: Int = 0)
    extends BlinkPortIO(b, inBW, outBW, new BallRsIssue(b), new BallRsComplete(b)) {
  val subRobReq = Decoupled(new SubRobRow(b))

  // MMIO metadata channels. Like bankRead/bankWrite, their number is a Ball
  // configuration property rather than a framework-wide fixed interface.
  val mmioRead  = Vec(mmioReadBW, Flipped(new MmioRead(b)))
  val mmioWrite = Vec(mmioWriteBW, Flipped(new MmioWrite(b)))
}
