package framework.acc.memdomain

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._

class LayoutConfig(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Bundle {
  val vbank_id       = Output(UInt(frontendParam.bank_id_len.W))
  val is_shared      = Output(Bool())
  val is_multi       = Output(Bool())
  val alloc          = Output(Bool())
  val transfer       = Output(Bool())
  val source_bank_id = Output(UInt(frontendParam.bank_id_len.W))
  val clear          = Output(Bool())
  val group_id       = Output(UInt(memParam.groupIdWidth.W))
  val thread_id      = Output(UInt(64.W))
}

import framework.acc.memdomain.{Request, Response}
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class Configer(val frontendParam: FrontendParam, val memParam: MemDomainParam, val permissionParam: PmpParams)
    extends Module {
  val ownerId = frontendParam.ownerId

  val rob_id_width = log2Up(frontendParam.rob_entries)

  @public
  val io = IO(new Bundle {
    val request   = Flipped(Decoupled(new Request(frontendParam, memParam, permissionParam)))
    val response  = Decoupled(new Response(frontendParam))
    val config    = Decoupled(new LayoutConfig(frontendParam, memParam))
    // A physical bank is still clearing; MSET.clear completes only after it drains.
    val clearBusy = Input(Bool())
  })

  val idle :: config :: clearWait :: resp :: Nil = Enum(4)

  val state              = RegInit(idle)
  val transfer_reg       = RegInit(false.B)
  val source_bank_id_reg = RegInit(0.U(frontendParam.bank_id_len.W))
  val alloc_reg          = RegInit(false.B)
  val is_shared_reg      = RegInit(false.B)
  val col_reg            = RegInit(0.U(memParam.groupCountWidth.W))
  val clear_reg          = RegInit(false.B)
  val vbank_id_reg       = RegInit(0.U(frontendParam.bank_id_len.W))
  val rob_id_reg         = RegInit(0.U(rob_id_width.W))
  val counter            = RegInit(0.U(memParam.groupCountWidth.W))

  io.config.bits.is_multi       := false.B
  io.config.bits.is_shared      := false.B
  io.config.bits.transfer       := false.B
  io.config.bits.source_bank_id := 0.U
  io.config.bits.alloc          := false.B
  io.config.bits.clear          := false.B
  io.config.bits.vbank_id       := 0.U(frontendParam.bank_id_len.W)
  io.config.bits.group_id       := 0.U
  io.config.bits.thread_id      := ownerId.U(64.W)
  io.config.valid               := false.B
  io.response.valid             := false.B
  io.response.bits              := 0.U.asTypeOf(io.response.bits)
  io.response.bits.fault        := 0.U.asTypeOf(io.response.bits.fault)
  io.response.bits.rob_id       := 0.U(rob_id_width.W)

  io.request.ready := state === idle

  when(state === idle) {
    when(io.request.valid) {
      when(io.request.fire) {
        val rawCol  = io.request.bits.cmd.special(9, 5)
        val alloc   = io.request.bits.cmd.special(10)
        val fullCol =
          if (memParam.sharedEnable) {
            Mux(
              io.request.bits.cmd.is_shared,
              memParam.sharedBankNum.U(col_reg.getWidth.W),
              memParam.bankNum.U(col_reg.getWidth.W)
            )
          } else {
            memParam.bankNum.U(col_reg.getWidth.W)
          }

        state              := config
        col_reg            := Mux(
          io.request.bits.cmd.transfer,
          1.U,
          Mux(alloc && rawCol === 0.U, fullCol, Mux(rawCol > 1.U, rawCol, 1.U))
        )
        transfer_reg       := io.request.bits.cmd.transfer
        source_bank_id_reg := io.request.bits.cmd.source_bank_id
        alloc_reg          := alloc
        clear_reg          := io.request.bits.cmd.clear
        is_shared_reg      := io.request.bits.cmd.is_shared
        vbank_id_reg       := io.request.bits.cmd.bank_id
        rob_id_reg         := io.request.bits.rob_id
        assert(
          !(io.request.bits.cmd.clear && io.request.bits.cmd.is_shared),
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
    io.config.bits.group_id       := counter(memParam.groupIdWidth - 1, 0)
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
    io.response.valid       := true.B
    io.response.bits.rob_id := rob_id_reg

    when(io.response.fire) {
      state   := idle
      counter := 0.U
    }
  }
}
