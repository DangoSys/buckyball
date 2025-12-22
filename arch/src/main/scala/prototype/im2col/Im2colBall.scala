package prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.im2col.Im2col
import prototype.im2col.configs.Im2colConfig

/**
 * Im2colBall - An Im2col computation Ball that complies with the Blink protocol
 */
@instantiable
class Im2colBall(config: Im2colConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  def Blink: Blink = io

  // Instantiate Im2col
  val im2colUnit: Instance[Im2col] = Instantiate(new Im2col(config))

  // Connect command interface
  im2colUnit.io.cmdReq <> io.cmdReq
  im2colUnit.io.cmdResp <> io.cmdResp

  // Connect Bank interface
  for (i <- 0 until parameter.numBanks) {
    im2colUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    im2colUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // Im2colBall uses overwrite mode
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> im2colUnit.io.status

  override lazy val desiredName = "Im2colBall"
}
