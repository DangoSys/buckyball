package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

class MemConfigerIO(val b: GlobalConfig) extends Bundle {
  val vbank_id = Output(UInt(8.W))
  val is_acc   = Output(Bool())
  val alloc    = Output(Bool())
}

@instantiable
class MemConfiger(val b: GlobalConfig) extends Module {

  val rob_id_width = log2Up(b.frontend.rob_entries)

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val config = Decoupled(new MemConfigerIO(b))
  })

  when(io.cmdReq.valid) {
    io.config.bits.is_acc   := io.cmdReq.bits.cmd.special(0)
    io.config.bits.alloc    := io.cmdReq.bits.cmd.special(1)
    io.config.bits.vbank_id := io.cmdReq.bits.cmd.bank_id
    io.config.valid         := true.B

    io.cmdResp.valid       := true.B
    io.cmdResp.bits.rob_id := io.cmdReq.bits.rob_id
  }.otherwise {
    io.config.bits.is_acc   := false.B
    io.config.bits.alloc    := false.B
    io.config.bits.vbank_id := 0.U(8.W)
    io.config.valid         := false.B

    io.cmdResp.valid       := false.B
    io.cmdResp.bits.rob_id := 0.U(rob_id_width.W)

  }
  io.cmdReq.ready := true.B
}
