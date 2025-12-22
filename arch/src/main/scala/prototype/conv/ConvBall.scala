package prototype.conv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.conv.Conv
import prototype.conv.configs.ConvConfig

/**
 * ConvBall - A Convolution computation Ball that complies with the Blink protocol
 * Behavior: Read input feature map and weights from Scratchpad, perform convolution using NVDLA CONV,
 * then write output feature map back to Scratchpad.
 */
@instantiable
class ConvBall(config: ConvConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate Conv computation unit
  private val convUnit: Instance[Conv] = Instantiate(new Conv(config))

  // Connect command interface
  convUnit.io.cmdReq <> io.cmdReq
  convUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    convUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    convUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // ConvBall uses overwrite mode for scratchpad

    // Accumulator write (for partial sums) - use accumulate mode
    // Note: ConvBall may need to write to accumulator banks with accumulate mode
    // For now, assuming all writes use overwrite mode, but accumulator writes should use accumulate
    // This needs to be determined based on Conv unit's actual behavior
  }

  // Pass through status signals
  io.status <> convUnit.io.status

  override lazy val desiredName: String = "ConvBall"
}
