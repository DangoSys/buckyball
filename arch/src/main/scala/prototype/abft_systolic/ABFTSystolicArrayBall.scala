package prototype.abft_systolic

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.abft_systolic.ABFTSystolicArray

/**
 * ABFTSystolicArrayBall - A systolic array Ball with ABFT support
 * Behavior: Read matrices A and B from Scratchpad, compute C = A * B with ABFT checks,
 * then write back to Scratchpad.
 */
class ABFTSystolicArrayBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate ABFT systolic array computation unit
  private val abftUnit = Module(new ABFTSystolicArray)

  // Connect command interface
  abftUnit.io.cmdReq <> io.cmdReq
  abftUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    abftUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    abftUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Accumulator write interface
  for (i <- 0 until b.acc_banks) {
    abftUnit.io.accWrite(i) <> io.accWrite(i).io
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
  io.status <> abftUnit.io.status

  override lazy val desiredName: String = "ABFTSystolicArrayBall"
}
