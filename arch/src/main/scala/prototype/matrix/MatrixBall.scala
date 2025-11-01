package prototype.matrix

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}
import prototype.matrix.BBFP_Control

/**
 * MatrixBall - A matrix computation Ball that complies with the Blink protocol
 */
class MatrixBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  // Instantiate BBFP_Control
  val matrixUnit = Module(new BBFP_Control)

  // Connect command interface
  matrixUnit.io.cmdReq <> io.cmdReq
  matrixUnit.io.cmdResp <> io.cmdResp

  // Set is_matmul_ws signal
  matrixUnit.io.is_matmul_ws := false.B  // TODO:

  // Connect SRAM read interface - MatrixBall needs to read data from scratchpad
  for (i <- 0 until b.sp_banks) {
    matrixUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect SRAM write interface - MatrixBall needs to write to scratchpad
  for (i <- 0 until b.sp_banks) {
    matrixUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Handle Accumulator read interface - MatrixBall does not read accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramReadIO), we need to drive req.valid, req.bits (outputs) and resp.ready (output)
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Connect Accumulator write interface - MatrixBall writes results to accumulator
  for (i <- 0 until b.acc_banks) {
    matrixUnit.io.accWrite(i) <> io.accWrite(i).io
    io.accWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Connect Status signals - directly obtained from internal unit
  io.status <> matrixUnit.io.status

  override lazy val desiredName = "MatrixBall"
}
