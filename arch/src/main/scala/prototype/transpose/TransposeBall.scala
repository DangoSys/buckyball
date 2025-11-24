package prototype.transpose

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.{Blink, BallRegist}
import prototype.transpose.PipelinedTransposer

/**
 * TransposeBall - A transpose computation Ball that complies with the Blink protocol
 */
class TransposeBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate PipelinedTransposer
  val transposeUnit = Module(new PipelinedTransposer)

  // Connect command interface
  transposeUnit.io.cmdReq <> io.cmdReq
  transposeUnit.io.cmdResp <> io.cmdResp

  // Connect SRAM read interface - Transpose needs to read data from scratchpad
  for (i <- 0 until b.sp_banks) {
    transposeUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect SRAM write interface - Transpose needs to write to scratchpad
  for (i <- 0 until b.sp_banks) {
    transposeUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Handle Accumulator read interface - Transpose does not read accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramReadIO), we need to drive req.valid, req.bits (outputs) and resp.ready (output)
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Handle Accumulator write interface - Transpose does not write accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramWriteIO), we need to drive req.valid and req.bits (outputs)
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> transposeUnit.io.status

  override lazy val desiredName = "TransposeBall"
}
