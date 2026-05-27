package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.cmd.rs.{MemRsComplete, MemRsIssue}
import framework.memdomain.backend.mmio.MmioAllocReq
import chisel3.experimental.hierarchy.{instantiable, public}

class MemConfigerIO(val b: GlobalConfig) extends Bundle {
  val vbank_id  = Output(UInt(8.W))
  val is_shared = Output(Bool())
  val is_multi  = Output(Bool())
  val alloc     = Output(Bool())
  val group_id  = Output(UInt(3.W))
  val hart_id   = Output(UInt(b.core.xLen.W))
}

@instantiable
class MemConfiger(val b: GlobalConfig) extends Module {

  val rob_id_width = log2Up(b.frontend.rob_entries)

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new MemRsIssue(b)))
    val cmdResp = Decoupled(new MemRsComplete(b))

    val config = Decoupled(new MemConfigerIO(b))
    val hartid = Input(UInt(b.core.xLen.W))

    // MMIO alloc/dealloc port
    val mmioAlloc = Valid(new MmioAllocReq(b))
  })

  val idle :: config :: resp :: Nil = Enum(3)
  val state                         = RegInit(idle)
  val alloc_reg                     = RegInit(false.B)
  val is_shared_reg                 = RegInit(false.B)
  val col_reg                       = RegInit(0.U(log2Up(b.memDomain.bankEntries).W))
  val vbank_id_reg                  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val rob_id_reg                    = RegInit(0.U(rob_id_width.W))
  val is_sub_reg                    = RegInit(false.B)
  val sub_rob_id_reg                = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val counter                       = RegInit(0.U(4.W))

  io.config.bits.is_multi    := false.B
  io.config.bits.is_shared   := false.B
  io.config.bits.alloc       := false.B
  io.config.bits.vbank_id    := 0.U(8.W)
  io.config.bits.group_id    := 0.U(3.W)
  io.config.bits.hart_id     := io.hartid
  io.config.valid            := false.B
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := 0.U(rob_id_width.W)
  io.cmdResp.bits.is_sub     := false.B
  io.cmdResp.bits.sub_rob_id := 0.U

  // MMIO alloc defaults
  io.mmioAlloc.valid          := false.B
  io.mmioAlloc.bits.main_bank := 0.U
  io.mmioAlloc.bits.mmio_addr := 0.U
  io.mmioAlloc.bits.size_rows := 0.U

  val isMmioSet = io.cmdReq.valid && io.cmdReq.bits.cmd.is_mmio_set
  io.cmdReq.ready := state === idle && (!isMmioSet || io.cmdResp.ready)

  when(state === idle) {
    when(io.cmdReq.valid) {
      // Check if this is mmio_set instruction
      when(io.cmdReq.bits.cmd.is_mmio_set) {
        // mmio_set: rs1[9:0]=main_bank, rs2[15:0]=mmio_addr, rs2[23:16]=size_rows
        val main_bank = io.cmdReq.bits.cmd.bank_id
        val mmio_addr = io.cmdReq.bits.cmd.special(15, 0)
        val size_rows = io.cmdReq.bits.cmd.special(23, 16)

        io.mmioAlloc.valid          := io.cmdReq.fire
        io.mmioAlloc.bits.main_bank := main_bank
        io.mmioAlloc.bits.mmio_addr := mmio_addr
        io.mmioAlloc.bits.size_rows := size_rows

        // Immediate response (no state transition)
        io.cmdResp.valid           := true.B
        io.cmdResp.bits.rob_id     := io.cmdReq.bits.rob_id
        io.cmdResp.bits.is_sub     := io.cmdReq.bits.is_sub
        io.cmdResp.bits.sub_rob_id := io.cmdReq.bits.sub_rob_id

      }.otherwise {
        when(io.cmdReq.fire) {
          state          := config
          col_reg        := Mux(io.cmdReq.bits.cmd.special(9, 5) > 1.U, io.cmdReq.bits.cmd.special(9, 5), 1.U)
          alloc_reg      := io.cmdReq.bits.cmd.special(10)
          is_shared_reg  := io.cmdReq.bits.cmd.is_shared
          vbank_id_reg   := io.cmdReq.bits.cmd.bank_id
          rob_id_reg     := io.cmdReq.bits.rob_id
          is_sub_reg     := io.cmdReq.bits.is_sub
          sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
        }
      }
    }

  }.elsewhen(state === config) {
    io.config.bits.is_multi  := col_reg > 1.U
    io.config.bits.is_shared := is_shared_reg
    io.config.bits.alloc     := alloc_reg
    io.config.bits.vbank_id  := vbank_id_reg
    io.config.bits.group_id  := counter
    io.config.valid          := true.B

    when(io.config.fire) {
      when(counter === col_reg - 1.U) {
        state := resp
      }.otherwise {
        counter := counter + 1.U
      }
    }
  }.otherwise {
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
