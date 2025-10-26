package examples.toy.balldomain.emptyball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{Blink, BallRegist}

class EmptyBall(id: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module with BallRegist {
  val io = IO(new Blink)
  val ballId = id.U

  def Blink: Blink = io

  io.cmdResp.valid := RegNext(io.cmdReq.valid)
  io.cmdResp.bits.rob_id := RegNext(io.cmdReq.bits.rob_id)
  io.cmdReq.ready := true.B

  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := false.B

    io.sramWrite(i).req.valid       := false.B
    io.sramWrite(i).req.bits.addr   := 0.U
    io.sramWrite(i).req.bits.data   := 0.U
    io.sramWrite(i).req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }

    // 处理Accumulator读接口 - Transpose不读accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramReadIO)，我们需要驱动req.valid, req.bits（输出）和resp.ready（输出）
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B
  }

  // 处理Accumulator写接口 - Transpose不写accumulator，所以tie off
  for (i <- 0 until b.acc_banks) {
    // 对于Flipped(SramWriteIO)，我们需要驱动req.valid和req.bits（输出）
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits := DontCare
  }
  io.status.ready := true.B
  io.status.valid := io.cmdResp.valid
  io.status.idle := false.B
  io.status.init := false.B
  io.status.running := false.B
  io.status.iter := 0.U
  io.status.complete := io.cmdResp.valid

}
