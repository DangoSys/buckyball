package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.backend.shared.SharedMemLayout
import framework.memdomain.frontend.cmd.rs.{MemRsComplete, MemRsIssue}
import chisel3.experimental.hierarchy.{instantiable, public}

class MemConfigerIO(val b: GlobalConfig) extends Bundle {
  val vbank_id       = Output(UInt(b.memDomain.vbankIdWidth.W))
  val is_shared      = Output(Bool())
  val is_multi       = Output(Bool())
  val alloc          = Output(Bool())
  val transfer       = Output(Bool())
  val source_bank_id = Output(UInt(b.memDomain.vbankIdWidth.W))
  // Zero the physical bank this allocation binds; the backend clears it locally.
  val clear          = Output(Bool())
  val group_id       = Output(UInt(b.memDomain.groupIdWidth.W))
  val hart_id        = Output(UInt(b.tile.xLen.W))
}

@instantiable
class MemConfiger(val b: GlobalConfig) extends Module {
  val rob_id_width = log2Up(b.frontend.rob_entries)

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp   = Decoupled(new MemRsComplete(b))
    val config    = Decoupled(new MemConfigerIO(b))
    val hartid    = Input(UInt(b.tile.xLen.W))
    // A backend bank is still clearing; MSET.clear completes only after it drains.
    val clearBusy = Input(Bool())
  })

  val idle :: config :: clearWait :: resp :: Nil = Enum(4)

  val state              = RegInit(idle)
  val transfer_reg       = RegInit(false.B)
  val source_bank_id_reg = RegInit(0.U(b.memDomain.vbankIdWidth.W))
  val alloc_reg          = RegInit(false.B)
  val is_shared_reg      = RegInit(false.B)
  val col_reg            = RegInit(0.U(b.memDomain.groupCountWidth.W))
  val clear_reg          = RegInit(false.B)
  val vbank_id_reg       = RegInit(0.U(b.memDomain.vbankIdWidth.W))
  val rob_id_reg         = RegInit(0.U(rob_id_width.W))
  val is_sub_reg         = RegInit(false.B)
  val sub_rob_id_reg     = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val counter            = RegInit(0.U(b.memDomain.groupCountWidth.W))

  io.config.bits.is_multi       := false.B
  io.config.bits.is_shared      := false.B
  io.config.bits.transfer       := false.B
  io.config.bits.source_bank_id := 0.U
  io.config.bits.alloc          := false.B
  io.config.bits.clear          := false.B
  io.config.bits.vbank_id       := 0.U(b.memDomain.vbankIdWidth.W)
  io.config.bits.group_id       := 0.U
  io.config.bits.hart_id        := io.hartid
  io.config.valid               := false.B
  io.cmdResp.valid              := false.B
  io.cmdResp.bits               := 0.U.asTypeOf(io.cmdResp.bits)
  io.cmdResp.bits.rob_id        := 0.U(rob_id_width.W)
  io.cmdResp.bits.is_sub        := false.B
  io.cmdResp.bits.sub_rob_id    := 0.U

  io.cmdReq.ready := state === idle

  when(state === idle) {
    when(io.cmdReq.valid) {
      when(io.cmdReq.fire) {
        val rawCol  = io.cmdReq.bits.cmd.special(9, 5)
        val alloc   = io.cmdReq.bits.cmd.special(10)
        val fullCol =
          if (b.memDomain.sharedEnable) {
            Mux(
              io.cmdReq.bits.cmd.is_shared,
              SharedMemLayout.totalBank(b).U(col_reg.getWidth.W),
              b.memDomain.bankNum.U(col_reg.getWidth.W)
            )
          } else {
            b.memDomain.bankNum.U(col_reg.getWidth.W)
          }

        state              := config
        col_reg            := Mux(
          io.cmdReq.bits.cmd.transfer,
          1.U,
          Mux(alloc && rawCol === 0.U, fullCol, Mux(rawCol > 1.U, rawCol, 1.U))
        )
        transfer_reg       := io.cmdReq.bits.cmd.transfer
        source_bank_id_reg := io.cmdReq.bits.cmd.source_bank_id
        alloc_reg          := alloc
        clear_reg          := io.cmdReq.bits.cmd.clear
        is_shared_reg      := io.cmdReq.bits.cmd.is_shared
        vbank_id_reg       := io.cmdReq.bits.cmd.bank_id
        rob_id_reg         := io.cmdReq.bits.rob_id
        is_sub_reg         := io.cmdReq.bits.is_sub
        sub_rob_id_reg     := io.cmdReq.bits.sub_rob_id
        assert(
          !(io.cmdReq.bits.cmd.clear && io.cmdReq.bits.cmd.is_shared),
          "MSET clear is currently supported for private banks only"
        )
      }
    }

  }.elsewhen(state === config) {
    io.config.bits.is_multi       := col_reg > 1.U
    io.config.bits.is_shared      := is_shared_reg
    io.config.bits.transfer       := transfer_reg
    io.config.bits.source_bank_id := source_bank_id_reg
    io.config.bits.alloc          := alloc_reg
    io.config.bits.clear          := clear_reg && alloc_reg
    io.config.bits.vbank_id       := vbank_id_reg
    io.config.bits.group_id       := counter(b.memDomain.groupIdWidth - 1, 0)
    io.config.valid               := true.B

    when(io.config.fire) {
      when(counter === col_reg - 1.U) {
        counter := 0.U
        state   := Mux(clear_reg && alloc_reg, clearWait, resp)
      }.otherwise {
        counter := counter + 1.U
      }
    }
  }.elsewhen(state === clearWait) {
    // Each bound bank started clearing when its allocation fired.
    when(!io.clearBusy)(state := resp)
  }.elsewhen(state === resp) {
    io.cmdResp.valid           := true.B
    io.cmdResp.bits.rob_id     := rob_id_reg
    io.cmdResp.bits.is_sub     := is_sub_reg
    io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

    when(io.cmdResp.fire) {
      state   := idle
      counter := 0.U
    }
  }
}
