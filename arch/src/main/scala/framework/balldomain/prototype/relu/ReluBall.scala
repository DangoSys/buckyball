package framework.balldomain.prototype.relu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallRegist, BlinkIO}
import framework.balldomain.prototype.relu.PipelinedRelu
import framework.top.GlobalConfig

/**
 * ReluBall - A ReLU computation Ball that complies with the Blink protocol.
 * Behavior: Read data from Scratchpad, perform element-wise ReLU (set negative values to 0),
 * then write back to Scratchpad.
 */
@instantiable
class ReluBall(val b: GlobalConfig) extends Module with BallRegist {
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "ReluBall")
    .getOrElse(throw new IllegalArgumentException("ReluBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def Blink: BlinkIO = io

  private val reluUnit: Instance[PipelinedRelu] = Instantiate(new PipelinedRelu(b))

  reluUnit.io.cmdReq <> io.cmdReq
  reluUnit.io.cmdResp <> io.cmdResp

  for (i <- 0 until inBW) {
    reluUnit.io.bankRead(i) <> io.bankRead(i).io
  }

  for (i <- 0 until outBW) {
    reluUnit.io.bankWrite(i) <> io.bankWrite(i).io
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id := io.cmdReq.bits.rob_id
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).io.req.bits.wmode := false.B
  }

  io.status <> reluUnit.io.status

}
