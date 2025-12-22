package prototype.nagisa.matmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.nagisa.matmul.marco

/**
 * marcoMatmulBall - A Compute-in-Memory Ball that complies with the Blink protocol
 * Behavior: Read operands from Scratchpad, perform marco computation (inspired by PiDRAM),
 * then write results back to Scratchpad.
 */
@instantiable
class marcoMatmulBall(parameter: BallDomainParam, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  @public
  val io     = IO(new Blink(parameter, parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen))
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate marco computation unit
  private val marcoUnit: Instance[marco] = Instantiate(new marco(parameter))

  // Connect command interface
  marcoUnit.io.cmdReq <> io.cmdReq
  marcoUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    marcoUnit.io.sramRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    marcoUnit.io.sramWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // marcoMatmulBall uses overwrite mode for scratchpad

    // Accumulator write (for partial sums) - use accumulate mode
    // Note: marco may need to write to accumulator banks with accumulate mode
    // For now, assuming all writes use overwrite mode, but accumulator writes should use accumulate
    // This needs to be determined based on marco unit's actual behavior
  }

  // Pass through status signals
  io.status <> marcoUnit.io.status

  override lazy val desiredName: String = "marcoMatmulBall"
}
