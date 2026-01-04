package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

class BlinkIO(b: GlobalConfig, inBW: Int, outBW: Int) extends Bundle with HasBallStatus {
  val status = new BallStatus()

  val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
  val cmdResp   = Decoupled(new BallRsComplete(b))
  val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
  val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
}
