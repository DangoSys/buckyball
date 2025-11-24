package prototype.vector

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.{Blink, BallRegist}
import prototype.vector.VecUnit

/**
 * VecBall - A vector computation Ball that complies with the Blink protocol
 */
class VecBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate VecUnit
  val vecUnit = Module(new VecUnit)

  // Connect command interface
  vecUnit.io.cmdReq <> io.cmdReq
  vecUnit.io.cmdResp <> io.cmdResp

  // Connect SRAM read interface - VecUnit needs to read data from scratchpad
  for (i <- 0 until b.sp_banks) {
    vecUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Handle SRAM write interface - VecUnit does not write to scratchpad, so tie off
  for (i <- 0 until b.sp_banks) {
    // For Flipped(SramWriteIO), we need to drive req.valid and req.bits (outputs)
    io.sramWrite(i).io.req.valid := false.B
    io.sramWrite(i).io.req.bits := DontCare
    io.sramWrite(i).rob_id := 0.U
  }

  // Handle Accumulator read interface - VecUnit does not read accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramReadIO), we need to drive req.valid, req.bits (outputs) and resp.ready (output)
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Connect Accumulator write interface - VecUnit writes results to accumulator
  for (i <- 0 until b.acc_banks) {
    vecUnit.io.accWrite(i) <> io.accWrite(i).io
    io.accWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> vecUnit.io.status

  override lazy val desiredName = "VecBall"
}
