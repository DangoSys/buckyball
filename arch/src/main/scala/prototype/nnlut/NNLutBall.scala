package prototype.nnlut

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.nnlut.NNLut
import prototype.nnlut.configs.NNLutConfig

/**
 * NNLutBall - A Neural Network Look-Up Table computation Ball that complies with the Blink protocol.
 * Behavior: Read data from Scratchpad, perform LUT lookup, then write back to Scratchpad.
 */
@instantiable
class NNLutBall(config: NNLutConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate NNLut computation unit
  private val nnlutUnit: Instance[NNLut] = Instantiate(new NNLut(config))

  // Connect command interface
  nnlutUnit.io.cmdReq <> io.cmdReq
  nnlutUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    nnlutUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    nnlutUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // NNLutBall uses overwrite mode
  }

  // Pass through status signals
  io.status <> nnlutUnit.io.status

  override lazy val desiredName: String = "NNLutBall"
}
