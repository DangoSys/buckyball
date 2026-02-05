package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MemConfiger(val b: GlobalConfig) extends Module {
  val rob_id_width = log2Up(b.frontend.rob_entries)

  // One bank line bytes
  private val line_bytes  = b.memDomain.bankWidth / 8
  // We pack/send 16B aligned beats to DMA
  private val align_bytes = 16

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val acc_config = Output(UInt(b.memDomain.bankNum.W))
  })

  val acc_config_reg = RegInit(0.U(b.memDomain.bankNum.W))

  when(io.cmdReq.valid) {
    acc_config_reg := io.cmdReq.bits.cmd.special(b.memDomain.bankNum - 1, 0)

    io.cmdResp.valid       := true.B
    io.cmdResp.bits.rob_id := io.cmdReq.bits.rob_id
  }.otherwise {
    io.cmdResp.valid       := false.B
    io.cmdResp.bits.rob_id := 0.U(rob_id_width.W)
  }
  io.cmdReq.ready := true.B
  io.acc_config   := acc_config_reg
}
