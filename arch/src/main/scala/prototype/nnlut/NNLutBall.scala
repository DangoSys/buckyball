package prototype.nnlut

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.blink.{Blink, BallRegist}
import prototype.nnlut.NNLut

/**
 * NNLutBall - A Neural Network Look-Up Table computation Ball that complies with the Blink protocol.
 * Behavior: Read data from Scratchpad, perform LUT lookup, then write back to Scratchpad.
 */
class NNLutBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate NNLut computation unit
  private val nnlutUnit = Module(new NNLut)

  // Connect command interface
  nnlutUnit.io.cmdReq <> io.cmdReq
  nnlutUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    nnlutUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    nnlutUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator read interface (NN-LUT does not access accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Accumulator write interface (NN-LUT does not write accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Pass through status signals
  io.status <> nnlutUnit.io.status

  override lazy val desiredName: String = "NNLutBall"
}
