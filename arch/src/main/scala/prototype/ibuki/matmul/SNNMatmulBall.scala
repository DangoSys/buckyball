package prototype.ibuki.matmul

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.ibuki.matmul.SNN

/**
 * SNNMatmulBall - A Spiking Neural Network computation Ball that complies with the Blink protocol.
 * Behavior: Read membrane potential from Scratchpad, apply LIF neuron model, generate spikes, then write back to Scratchpad.
 */
class SNNMatmulBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate SNN computation unit
  private val snnUnit = Module(new SNN)

  // Connect command interface
  snnUnit.io.cmdReq <> io.cmdReq
  snnUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    snnUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    snnUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator read interface (SNN does not access accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Accumulator write interface (SNN does not write accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Pass through status signals
  io.status <> snnUnit.io.status

  override lazy val desiredName: String = "SNNMatmulBall"
}
