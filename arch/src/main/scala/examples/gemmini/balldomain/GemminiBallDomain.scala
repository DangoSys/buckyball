package examples.gemmini.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.PostGDCmd
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.rocket.RoCCResponseBB

/**
 * GemminiBallDomain: Gemmini的Ball域
 * 目前暂时为空实现，后续添加计算Ball
 */
class GemminiBallDomainIO(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val gDecoderIn = Flipped(Decoupled(new PostGDCmd))

  val sramRead  = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
  val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
  val accRead   = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
  val accWrite  = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

  val roccResp = Decoupled(new RoCCResponseBB()(p))
  val busy = Output(Bool())
}

class GemminiBallDomain(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new GemminiBallDomainIO)

  // 暂时的简单实现：直接接受所有命令但不做任何事情
  io.gDecoderIn.ready := true.B

  // SRAM接口暂时不使用 - 注意Flipped后方向，我们只应该驱动req和resp.ready
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid := false.B
    io.sramRead(i).req.bits := DontCare
    io.sramRead(i).resp.ready := true.B

    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits := DontCare
  }

  for (i <- 0 until b.acc_banks) {
    io.accRead(i).req.valid := false.B
    io.accRead(i).req.bits := DontCare
    io.accRead(i).resp.ready := true.B

    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits := DontCare
  }

  // RoCC响应：暂时不返回任何响应
  io.roccResp.valid := false.B
  io.roccResp.bits := DontCare

  // 不忙
  io.busy := false.B

  override lazy val desiredName = "GemminiBallDomain"
}
