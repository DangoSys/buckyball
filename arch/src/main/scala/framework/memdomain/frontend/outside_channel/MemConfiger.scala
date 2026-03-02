package framework.memdomain.frontend.outside_channel

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd_channel.rs.{MemRsComplete, MemRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

class MemConfigerIO(val b: GlobalConfig) extends Bundle {
  val vbank_id = Output(UInt(8.W))
  val is_multi = Output(Bool())
  val alloc    = Output(Bool())
  val group_id = Output(UInt(3.W))
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
  val alloc_reg             = RegInit(false.B)
  val row_reg               = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val col_reg               = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val vbank_id_reg          = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val rob_id_reg            = RegInit(0.U(rob_id_width.W))
  val counter               = RegInit(0.U(4.W))

  io.config.bits.is_multi := false.B
  io.config.bits.alloc    := false.B
  io.config.bits.vbank_id := 0.U(8.W)
  io.config.bits.group_id := 0.U(3.W)
  io.config.valid         := false.B
  io.cmdResp.valid        := false.B
  io.cmdResp.bits.rob_id  := 0.U(rob_id_width.W)

  when(state === idle) {
    when(io.cmdReq.valid) {
      when(io.cmdReq.bits.cmd.special(9, 5) > 1.U) { //is multi bank
        state        := config
        row_reg      := io.cmdReq.bits.cmd.special(4, 0)
        col_reg      := io.cmdReq.bits.cmd.special(9, 5)
        alloc_reg    := io.cmdReq.bits.cmd.special(10)
        vbank_id_reg := io.cmdReq.bits.cmd.bank_id
        rob_id_reg   := io.cmdReq.bits.rob_id

      }.otherwise { //not multi bank
        io.config.bits.alloc    := io.cmdReq.bits.cmd.special(10)
        io.config.bits.vbank_id := io.cmdReq.bits.cmd.bank_id
        io.config.valid         := true.B

        io.cmdResp.valid       := true.B
        io.cmdResp.bits.rob_id := io.cmdReq.bits.rob_id
      }
    }

  }.otherwise {
    when(counter < col_reg) {
      io.config.bits.is_multi := true.B
      io.config.bits.alloc    := alloc_reg
      io.config.bits.vbank_id := vbank_id_reg
      io.config.bits.group_id := counter
      io.config.valid         := true.B
      counter                 := counter + 1.U
    }.otherwise {
      state                  := idle
      counter                := 0.U
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
    }
  }
  io.cmdReq.ready := state === idle
}
