package prototype.nagisa.matmul

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.{Blink, BallRegist}
import prototype.nagisa.matmul.marco

/**
 * marcoMatmulBall - A Compute-in-Memory Ball that complies with the Blink protocol
 * Behavior: Read operands from Scratchpad, perform marco computation (inspired by PiDRAM),
 * then write results back to Scratchpad.
 */
class marcoMatmulBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate marco computation unit
  private val marcoUnit = Module(new marco)

  // Connect command interface
  marcoUnit.io.cmdReq <> io.cmdReq
  marcoUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    marcoUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    marcoUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Accumulator write interface (for partial sums in marco operations)
  for (i <- 0 until b.acc_banks) {
    marcoUnit.io.accWrite(i) <> io.accWrite(i).io
    io.accWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator read interface (not used, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Pass through status signals
  io.status <> marcoUnit.io.status

  override lazy val desiredName: String = "marcoMatmulBall"
}
