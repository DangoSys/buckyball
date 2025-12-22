package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.matrix.BBFP_Control
import prototype.matrix.configs.MatrixConfig

/**
 * MatrixBall - A matrix computation Ball that complies with the Blink protocol
 */
@instantiable
class MatrixBall(config: MatrixConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  def Blink: Blink = io

  // Instantiate BBFP_Control
  val matrixUnit: Instance[BBFP_Control] = Instantiate(new BBFP_Control(parameter))

  // Connect command interface
  matrixUnit.io.cmdReq <> io.cmdReq
  matrixUnit.io.cmdResp <> io.cmdResp

  // Set is_matmul_ws signal
  matrixUnit.io.is_matmul_ws := false.B // TODO:

  // Connect Bank interface
  for (i <- 0 until parameter.numBanks) {
    matrixUnit.io.sramRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    matrixUnit.io.sramWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // MatrixBall uses overwrite mode
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> matrixUnit.io.status

  override lazy val desiredName = "MatrixBall"
}
