package prototype.transfer

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.blink.{Blink, BallRegist}
import prototype.transfer.Transfer

// TransferBall - A data transfer Ball that complies with the Blink protocol.
// Behavior: Read data from Scratchpad and write it back without modification.

class TransferBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters)
    extends Module
    with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  // Satisfy BallRegist requirements
  def Blink: Blink = io

  // Instantiate Transfer computation unit
  private val transferUnit = Module(new Transfer[UInt])

  // Connect command interface
  transferUnit.io.cmdReq <> io.cmdReq
  transferUnit.io.cmdResp <> io.cmdResp

  // Connect Scratchpad SRAM read/write interface
  for (i <- 0 until b.sp_banks) {
    transferUnit.io.sramRead(i) <> io.sramRead(i).io
    io.sramRead(i).rob_id := io.cmdReq.bits.rob_id
    transferUnit.io.sramWrite(i) <> io.sramWrite(i).io
    io.sramWrite(i).rob_id := io.cmdReq.bits.rob_id
  }

  // Accumulator read interface (Transfer does not access accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Accumulator write interface (Transfer does not write accumulator, tie-off)
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }

  // Pass through status signals
  io.status <> transferUnit.io.status

  override lazy val desiredName: String = "TransferBall"
}
