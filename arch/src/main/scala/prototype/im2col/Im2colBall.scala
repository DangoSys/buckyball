package prototype.im2col

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.im2col.Im2col

/**
 * Im2colBall - An Im2col computation Ball that complies with the Blink protocol
 */
class Im2colBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate Im2col
  val im2colUnit = Module(new Im2col)

  // Connect command interface
  im2colUnit.io.cmdReq <> io.cmdReq
  im2colUnit.io.cmdResp <> io.cmdResp

  // Connect SRAM read interface - Im2col needs to read data from scratchpad
  for (i <- 0 until b.sp_banks) {
    im2colUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect SRAM write interface - Im2col needs to write to scratchpad
  for (i <- 0 until b.sp_banks) {
    im2colUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Handle Accumulator read interface - Im2col does not read accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramReadIO), we need to drive req.valid, req.bits (outputs) and resp.ready (output)
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Handle Accumulator write interface - Im2col does not write accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramWriteIO), we need to drive req.valid and req.bits (outputs)
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> im2colUnit.io.status

  override lazy val desiredName = "Im2colBall"
}
