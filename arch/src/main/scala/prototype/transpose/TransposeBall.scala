package prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.transpose.PipelinedTransposer
import prototype.transpose.configs.TransposeConfig

/**
 * TransposeBall - A transpose computation Ball that complies with the Blink protocol
 */
@instantiable
class TransposeBall(config: TransposeConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  def Blink: Blink = io

  // Instantiate PipelinedTransposer
  val transposeUnit: Instance[PipelinedTransposer] = Instantiate(new PipelinedTransposer(config))

  // Connect command interface
  transposeUnit.io.cmdReq <> io.cmdReq
  transposeUnit.io.cmdResp <> io.cmdResp

  // Connect Bank interface
  for (i <- 0 until parameter.numBanks) {
    transposeUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    transposeUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // TransposeBall uses overwrite mode
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> transposeUnit.io.status

  override lazy val desiredName = "TransposeBall"
}
