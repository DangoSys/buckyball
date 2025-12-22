package prototype.abft

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.abft.ABFTSystolicArray
import prototype.abft.configs.ABFTConfig

/**
 * ABFTSystolicArrayBall - A systolic array Ball with ABFT support
 * Behavior: Read matrices A and B from Scratchpad, compute C = A * B with ABFT checks,
 * then write back to Scratchpad.
 */
@instantiable
class ABFTSystolicArrayBall(config: ABFTConfig, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  val parameter = config.ballParam
  @public
  val io        = IO(new Blink(parameter, config.bankEntries, config.bankWidth, config.bankMaskLen))
  val ballId    = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate ABFT systolic array computation unit
  private val abftUnit: Instance[ABFTSystolicArray] = Instantiate(new ABFTSystolicArray(config))

  // Connect command interface
  abftUnit.io.cmdReq <> io.cmdReq
  abftUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    abftUnit.io.bankRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    abftUnit.io.bankWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // ABFTSystolicArrayBall uses overwrite mode for scratchpad

    // Accumulator write (for partial sums) - use accumulate mode
    abftUnit.io.bankWriteAcc(i) <> io.bankWrite(i).io
    // Note: bankWriteAcc uses accumulate mode, but we connect to the same bankWrite interface
    // The wmode should be set to true for accumulator writes
  }

  // Pass through status signals
  io.status <> abftUnit.io.status

  override lazy val desiredName: String = "ABFTSystolicArrayBall"
}
