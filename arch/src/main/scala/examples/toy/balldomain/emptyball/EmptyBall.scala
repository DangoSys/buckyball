package examples.toy.balldomain.emptyball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.blink.{Blink, BallRegist}

class EmptyBall(id: Int)(implicit b: CustomBuckyballConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  io.cmdResp.valid := RegNext(io.cmdReq.valid)
  io.cmdResp.bits.rob_id := RegNext(io.cmdReq.bits.rob_id)
  io.cmdReq.ready := true.B

  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).io.req.valid        := false.B
    io.sramRead(i).io.req.bits.addr    := 0.U
    io.sramRead(i).io.req.bits.fromDMA := false.B
    io.sramRead(i).io.resp.ready       := false.B
    io.sramRead(i).rob_id              := 0.U

    io.sramWrite(i).io.req.valid       := false.B
    io.sramWrite(i).io.req.bits.addr   := 0.U
    io.sramWrite(i).io.req.bits.data   := 0.U
    io.sramWrite(i).io.req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
    io.sramWrite(i).rob_id             := 0.U
  }

    // Handle Accumulator read interface - EmptyBall does not read accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramReadIO), we need to drive req.valid, req.bits (outputs) and resp.ready (output)
    io.accRead(i).io.req.valid := false.B
    io.accRead(i).io.req.bits := DontCare
    io.accRead(i).io.resp.ready := true.B
    io.accRead(i).rob_id := 0.U
  }

  // Handle Accumulator write interface - EmptyBall does not write accumulator, so tie off
  for (i <- 0 until b.acc_banks) {
    // For Flipped(SramWriteIO), we need to drive req.valid and req.bits (outputs)
    io.accWrite(i).io.req.valid := false.B
    io.accWrite(i).io.req.bits := DontCare
    io.accWrite(i).rob_id := 0.U
  }
  io.status.ready := true.B
  io.status.valid := io.cmdResp.valid
  io.status.idle := false.B
  io.status.init := false.B
  io.status.running := false.B
  io.status.iter := 0.U
  io.status.complete := io.cmdResp.valid

}
