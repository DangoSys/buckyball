package prototype.transfer

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.transfer.Transfer
import prototype.transfer.configs.TransferConfig

// TransferBall - A data transfer Ball that complies with the Blink protocol.
// Behavior: Read data from Scratchpad and write it back without modification.

@instantiable
class TransferBall(config: TransferConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate Transfer computation unit
  private val transferUnit: Instance[Transfer] = Instantiate(new Transfer(config))

  // Connect command interface
  transferUnit.io.cmdReq <> io.cmdReq
  transferUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    transferUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id             := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id            := i.U
    transferUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // TransferBall uses overwrite mode
  }

  // Pass through status signals
  io.status <> transferUnit.io.status

  override lazy val desiredName: String = "TransferBall"
}
