package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

class MemConfigerIO(val b: GlobalConfig) extends Bundle {
  val vbank_id     = Output(UInt(8.W))
  val is_acc       = Output(Bool())
  val alloc        = Output(Bool())
  val acc_group_id = Output(UInt(3.W))
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

  val idle :: config :: Nil = Enum(2)
  val state                 = RegInit(idle)
  val is_acc_reg            = RegInit(false.B)
  val alloc_reg             = RegInit(false.B)
  val vbank_id_reg          = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val rob_id_reg            = RegInit(0.U(rob_id_width.W))
  val counter               = RegInit(0.U(4.W))

  io.config.bits.is_acc       := false.B
  io.config.bits.alloc        := false.B
  io.config.bits.vbank_id     := 0.U(8.W)
  io.config.bits.acc_group_id := 0.U(3.W)
  io.config.valid             := false.B
  io.cmdResp.valid            := false.B
  io.cmdResp.bits.rob_id      := 0.U(rob_id_width.W)

  when(state === idle) {
    when(io.cmdReq.valid) {
      when(io.cmdReq.bits.cmd.special(0)) { //is acc
        state        := config
        is_acc_reg   := io.cmdReq.bits.cmd.special(0)
        alloc_reg    := io.cmdReq.bits.cmd.special(1)
        vbank_id_reg := io.cmdReq.bits.cmd.bank_id
        rob_id_reg   := io.cmdReq.bits.rob_id

      }.otherwise { //not acc
        io.config.bits.alloc    := io.cmdReq.bits.cmd.special(1)
        io.config.bits.vbank_id := io.cmdReq.bits.cmd.bank_id
        io.config.valid         := true.B

        io.cmdResp.valid       := true.B
        io.cmdResp.bits.rob_id := io.cmdReq.bits.rob_id
      }
    }

  }.otherwise {
    when(counter < 4.U) {
      io.config.bits.is_acc       := is_acc_reg
      io.config.bits.alloc        := alloc_reg
      io.config.bits.vbank_id     := vbank_id_reg
      io.config.bits.acc_group_id := counter
      io.config.valid             := true.B
      counter                     := counter + 1.U
    }.otherwise {
      state                  := idle
      counter                := 0.U
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
    }
  }
  io.cmdReq.ready := state === idle
}
