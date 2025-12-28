package framework.balldomain.prototype.relu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import framework.balldomain.blink.{BallRegist, BlinkIO}
import framework.balldomain.prototype.relu.PipelinedRelu
import framework.top.GlobalConfig

/**
 * ReluBall - A ReLU computation Ball that complies with the Blink protocol.
 * Behavior: Read data from Scratchpad, perform element-wise ReLU (set negative values to 0),
 * then write back to Scratchpad.
 */
@instantiable
class ReluBall(val b: GlobalConfig, id: Int) extends Module with BallRegist {
  @public
  val io     = IO(new BlinkIO(b))
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: BlinkIO = io

  // Instantiate PipelinedRelu computation unit
  private val reluUnit: Instance[PipelinedRelu] = Instantiate(new PipelinedRelu(b))

  // Connect command interface
  reluUnit.io.cmdReq <> io.cmdReq
  reluUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until b.memDomain.bankNum) {
    reluUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id             := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id            := i.U
    reluUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // ReluBall uses overwrite mode
  }

  // Pass through status signals
  io.status <> reluUnit.io.status

}
