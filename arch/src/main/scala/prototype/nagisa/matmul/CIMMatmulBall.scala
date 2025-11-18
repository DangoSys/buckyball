package prototype.nagisa.matmul

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.nagisa.matmul.CIM

/**
 * CIMMatmulBall - A Compute-in-Memory Ball that complies with the Blink protocol
 * Behavior: Read operands from Scratchpad, perform CIM computation (inspired by PiDRAM),
 * then write results back to Scratchpad.
 */
class CIMMatmulBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate CIM computation unit
  private val cimUnit = Module(new CIM)

  // Connect command interface
  cimUnit.io.cmdReq <> io.cmdReq
  cimUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    cimUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    cimUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Accumulator write interface (for partial sums in CIM operations)
  for (i <- 0 until b.acc_banks) {
    cimUnit.io.accWrite(i) <> io.accWrite(i).io
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
  io.status <> cimUnit.io.status

  override lazy val desiredName: String = "CIMMatmulBall"
}
