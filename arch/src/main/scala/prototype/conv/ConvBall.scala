package prototype.conv

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.conv.Conv

/**
 * ConvBall - A Convolution computation Ball that complies with the Blink protocol
 * Behavior: Read input feature map and weights from Scratchpad, perform convolution using NVDLA CONV,
 * then write output feature map back to Scratchpad.
 */
class ConvBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate Conv computation unit
  private val convUnit = Module(new Conv)

  // Connect command interface
  convUnit.io.cmdReq <> io.cmdReq
  convUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    convUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    convUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Accumulator write interface (for partial sums in convolution)
  for (i <- 0 until b.acc_banks) {
    convUnit.io.accWrite(i) <> io.accWrite(i).io
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
  io.status <> convUnit.io.status

  override lazy val desiredName: String = "ConvBall"
}
