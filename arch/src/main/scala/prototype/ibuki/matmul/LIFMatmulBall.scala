package prototype.ibuki.matmul

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.{Blink, BallRegist}
import prototype.ibuki.matmul.LIF


class LIFMatmulBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate LIF computation unit
  private val lifUnit = Module(new LIF)

  // Connect command interface
  lifUnit.io.cmdReq <> io.cmdReq
  lifUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    lifUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    lifUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator read interface (LIF does not access accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Accumulator write interface (LIF does not write accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Pass through status signals
  io.status <> lifUnit.io.status

  override lazy val desiredName: String = "LIFMatmulBall"
}
