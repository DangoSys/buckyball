package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.memdomain.backend.MemRequestIO

@instantiable
class AccPipe(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val sramRead  = Flipped(new SramReadIO(b))
    val sramWrite = Flipped(new SramWriteIO(b))

    val mem_req  = Flipped(new MemRequestIO(b))
    val is_multi = Input(Bool())

    val busy     = Output(Bool())
    val group_id = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val bank_id  = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val hart_id  = Output(UInt(b.core.xLen.W))
  })

  // Each group has its own physical bank, so no address shifting is needed.
  // The previous is_multi shift (addr >> 2) was incorrect: it caused mvout reads
  // to access wrong physical addresses while matmul writes used unshifted addresses.

  //group_id output
  val group_id_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  io.group_id := group_id_reg

  //Bank_id output
  val bank_id_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  io.bank_id := bank_id_reg
  val hart_id_reg = RegInit(0.U(b.core.xLen.W))
  io.hart_id := hart_id_reg

  val rd_queued   = RegInit(false.B)
  val rd_inflight = RegInit(false.B)
  val rd_hold     = RegInit(false.B)
  val rd_addr     = Reg(UInt(log2Ceil(b.memDomain.bankEntries).W))
  val rd_data_reg = RegInit(0.U(b.memDomain.bankWidth.W))
  val wr_queued   = RegInit(false.B)
  val wr_inflight = RegInit(false.B)
  val wr_hold     = RegInit(false.B)
  val wr_addr     = Reg(UInt(log2Ceil(b.memDomain.bankEntries).W))
  val wr_data     = Reg(UInt(b.memDomain.bankWidth.W))
  val wr_mask     = Reg(Vec(b.memDomain.bankMaskLen, Bool()))
  val wr_ok_reg   = RegInit(false.B)

  val canStart    = !rd_queued && !rd_inflight && !rd_hold && !wr_queued && !wr_inflight && !wr_hold
  val hasWriteReq = io.mem_req.write.req.valid
  val wrReq       = wr_queued
  val rdReq       = rd_queued

  io.sramRead.req.valid     := rdReq
  io.sramRead.req.bits.addr := rd_addr
  io.sramRead.resp.ready    := !rd_hold

  io.sramWrite.req.valid     := wrReq
  io.sramWrite.req.bits.addr := wr_addr
  io.sramWrite.req.bits.data := wr_data
  io.sramWrite.req.bits.mask := wr_mask
  io.sramWrite.resp.ready    := !wr_hold

  io.mem_req.read.req.ready      := canStart && !hasWriteReq
  io.mem_req.read.resp.valid     := rd_hold
  io.mem_req.read.resp.bits.data := rd_data_reg

  io.mem_req.write.req.ready    := canStart
  io.mem_req.write.resp.valid   := wr_hold
  io.mem_req.write.resp.bits.ok := wr_ok_reg

  when(io.mem_req.read.req.fire) {
    rd_queued    := true.B
    rd_addr      := io.mem_req.read.req.bits.addr
    group_id_reg := io.mem_req.group_id
    bank_id_reg  := io.mem_req.bank_id
    hart_id_reg  := io.mem_req.hart_id
  }
  when(io.sramRead.req.fire) {
    rd_queued   := false.B
    rd_inflight := true.B
  }
  when(io.sramRead.resp.fire) {
    rd_inflight := false.B
    rd_hold     := true.B
    rd_data_reg := io.sramRead.resp.bits.data
  }
  when(rd_hold && io.mem_req.read.resp.ready) {
    rd_hold := false.B
  }

  when(io.mem_req.write.req.fire) {
    wr_queued    := true.B
    wr_addr      := io.mem_req.write.req.bits.addr
    wr_data      := io.mem_req.write.req.bits.data
    wr_mask      := io.mem_req.write.req.bits.mask
    group_id_reg := io.mem_req.group_id
    bank_id_reg  := io.mem_req.bank_id
    hart_id_reg  := io.mem_req.hart_id
  }
  when(io.sramWrite.req.fire) {
    wr_queued   := false.B
    wr_inflight := true.B
  }
  when(io.sramWrite.resp.fire) {
    wr_inflight := false.B
    wr_hold     := true.B
    wr_ok_reg   := io.sramWrite.resp.bits.ok
  }
  when(wr_hold && io.mem_req.write.resp.ready) {
    wr_hold := false.B
  }

  io.busy := rd_queued || rd_inflight || rd_hold || wr_queued || wr_inflight || wr_hold
}
