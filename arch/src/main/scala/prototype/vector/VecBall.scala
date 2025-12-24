package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.vector.VecUnit
import prototype.vector.configs.VecConfig

/**
 * VecBall - A vector computation Ball that complies with the Blink protocol
 */
@instantiable
class VecBall(config: VecConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  def Blink: Blink = io

  // Instantiate VecUnit
  val vecUnit: Instance[VecUnit] = Instantiate(new VecUnit(parameter))

  // Connect command interface
  vecUnit.io.cmdReq <> io.cmdReq
  vecUnit.io.cmdResp <> io.cmdResp

  // Connect Bank interface
  for (i <- 0 until parameter.numBanks) {
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

  override lazy val desiredName = "VecBall"
}
