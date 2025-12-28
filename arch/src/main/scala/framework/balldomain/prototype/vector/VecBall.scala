package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallRegist, BlinkIO}
import framework.balldomain.prototype.vector.VecUnit
import framework.top.GlobalConfig

/**
 * VecBall - A vector computation Ball that complies with the Blink protocol
 */
@instantiable
class VecBall(val b: GlobalConfig, id: Int) extends Module with BallRegist {
  @public
  val io     = IO(new BlinkIO(b))
  val ballId = id.U

  def Blink: BlinkIO = io

  // Instantiate VecUnit
  val vecUnit: Instance[VecUnit] = Instantiate(new VecUnit(b))

  // Connect command interface
  vecUnit.io.cmdReq <> io.cmdReq
  vecUnit.io.cmdResp <> io.cmdResp

  // Connect Bank interface
  for (i <- 0 until b.memDomain.bankNum) {
    vecUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    // VecUnit uses bankWrite for writes (accumulate mode)
    vecUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := true.B // VecBall uses accumulate mode for writes
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> vecUnit.io.status

}
