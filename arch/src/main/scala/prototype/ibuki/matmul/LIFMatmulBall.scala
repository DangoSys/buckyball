package prototype.ibuki.matmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}
import prototype.ibuki.matmul.LIF

@instantiable
class LIFMatmulBall(parameter: BallDomainParam, id: Int)(implicit p: Parameters) extends Module with BallRegist {
  @public
  val io     = IO(new Blink(parameter, parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen))
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate LIF computation unit
  private val lifUnit: Instance[LIF] = Instantiate(new LIF(parameter))

  // Connect command interface
  lifUnit.io.cmdReq <> io.cmdReq
  lifUnit.io.cmdResp <> io.cmdResp

  // Connect Bank read/write interface
  for (i <- 0 until parameter.numBanks) {
    lifUnit.io.sramRead(i) <> io.bankRead(i).io
    io.bankRead(i).rob_id  := io.cmdReq.bits.rob_id
    io.bankRead(i).bank_id := i.U

    lifUnit.io.sramWrite(i) <> io.bankWrite(i).io
    io.bankWrite(i).rob_id            := io.cmdReq.bits.rob_id
    io.bankWrite(i).bank_id           := i.U
    io.bankWrite(i).io.req.bits.wmode := false.B // LIFMatmulBall uses overwrite mode
  }

  // Pass through status signals
  io.status <> lifUnit.io.status

  override lazy val desiredName: String = "LIFMatmulBall"
}
