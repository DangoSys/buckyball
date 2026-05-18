package framework.balldomain.prototype.hadamard

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.{BlinkIO, BallStatus}
import framework.top.GlobalConfig

@instantiable
class HadamardUnit(val b: GlobalConfig) extends Module {
  val config = b.ballDomain.ballIdMappings.find(_.ballName == "HadamardBall").get
  val inBW  = config.inBW
  val outBW = config.outBW

  @public
  val io = IO(new BlinkIO(b,inBW,outBW))


  val vecA = io.bankRead(0).data
  val vecB = io.bankRead(1).data
  val result = vecA.zip(vecB).map {case (a,b) => a*b}

  io.bankWrite(0).data := result

  io.status.idle    := true.B
  io.status.running := false.B
  io.status.error   := 0.U
}