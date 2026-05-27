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
    val group_id = Output(UInt(3.W))
    val bank_id  = Output(UInt(log2Up(b.memDomain.bankNum).W))
  })

  val idle :: accReadResp :: accWriteReq :: accWriteResp :: Nil = Enum(4)
  val state                                                     = RegInit(idle)

  // Each group has its own physical bank, so no address shifting is needed.
  // The previous is_multi shift (addr >> 2) was incorrect: it caused mvout reads
  // to access wrong physical addresses while matmul writes used unshifted addresses.

  //group_id output
  val group_id_reg = RegInit(0.U(3.W))
  io.group_id := group_id_reg

  //Bank_id output
  val bank_id_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  io.bank_id := bank_id_reg

  val acc_data_reg = RegInit(0.U(b.memDomain.bankWidth.W))
  val acc_mask_reg = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))
  val acc_addr_reg = RegInit(0.U(b.memDomain.memAddrLen.W))
  val old_data_reg = RegInit(0.U(b.memDomain.bankWidth.W))
  val rd_hold      = RegInit(false.B)
  val rd_data_reg  = RegInit(0.U(b.memDomain.bankWidth.W))
  val wr_hold      = RegInit(false.B)
  val wr_ok_reg    = RegInit(false.B)

  def laneAdd(a: UInt, old: UInt): UInt = {
    val lanes = b.memDomain.bankWidth / 32
    Cat((0 until lanes).reverse.map { i =>
      val hi = (i + 1) * 32 - 1
      val lo = i * 32
      a(hi, lo) + old(hi, lo)
    })
  }

  val canStart    = state === idle && !rd_hold && !wr_hold &&
    !io.sramRead.resp.valid && !io.sramWrite.resp.valid
  val hasWriteReq = io.mem_req.write.req.valid
  val accReq      = canStart && hasWriteReq && io.mem_req.write.req.bits.wmode
  val wrReq       = canStart && hasWriteReq && !io.mem_req.write.req.bits.wmode
  val rdReq       = canStart && !hasWriteReq && io.mem_req.read.req.valid

  io.sramRead.req.valid     := rdReq || accReq
  io.sramRead.req.bits.addr := Mux(accReq, io.mem_req.write.req.bits.addr, io.mem_req.read.req.bits.addr)
  io.sramRead.resp.ready    := state === idle || state === accReadResp

  io.sramWrite.req.valid      := wrReq || state === accWriteReq
  io.sramWrite.req.bits.addr  := Mux(state === accWriteReq, acc_addr_reg, io.mem_req.write.req.bits.addr)
  io.sramWrite.req.bits.data  := Mux(
    state === accWriteReq,
    laneAdd(acc_data_reg, old_data_reg),
    io.mem_req.write.req.bits.data
  )
  io.sramWrite.req.bits.mask  := Mux(state === accWriteReq, acc_mask_reg, io.mem_req.write.req.bits.mask)
  io.sramWrite.req.bits.wmode := Mux(state === accWriteReq, true.B, io.mem_req.write.req.bits.wmode)
  io.sramWrite.resp.ready     := true.B

  io.mem_req.read.req.ready      := canStart && !hasWriteReq && io.sramRead.req.ready
  io.mem_req.read.resp.valid     := rd_hold || (state === idle && io.sramRead.resp.valid)
  io.mem_req.read.resp.bits.data := Mux(rd_hold, rd_data_reg, io.sramRead.resp.bits.data)

  io.mem_req.write.req.ready    := Mux(
    io.mem_req.write.req.bits.wmode,
    canStart && io.sramRead.req.ready,
    canStart && io.sramWrite.req.ready
  )
  io.mem_req.write.resp.valid   := wr_hold || io.sramWrite.resp.valid
  io.mem_req.write.resp.bits.ok := Mux(wr_hold, wr_ok_reg, io.sramWrite.resp.bits.ok)

  when(rd_hold) {
    when(io.mem_req.read.resp.ready) {
      rd_hold := false.B
    }
  }.elsewhen(state === idle && io.sramRead.resp.valid && !io.mem_req.read.resp.ready) {
    rd_hold     := true.B
    rd_data_reg := io.sramRead.resp.bits.data
  }

  when(wr_hold) {
    when(io.mem_req.write.resp.ready) {
      wr_hold := false.B
    }
  }.elsewhen(io.sramWrite.resp.valid && !io.mem_req.write.resp.ready) {
    wr_hold   := true.B
    wr_ok_reg := io.sramWrite.resp.bits.ok
  }

  when(state === idle) {
    when(io.mem_req.read.req.fire) {
      group_id_reg := io.mem_req.group_id
      bank_id_reg  := io.mem_req.bank_id
    }
    when(io.mem_req.write.req.fire) {
      group_id_reg := io.mem_req.group_id
      bank_id_reg  := io.mem_req.bank_id
      when(io.mem_req.write.req.bits.wmode) {
        acc_data_reg := io.mem_req.write.req.bits.data
        acc_mask_reg := io.mem_req.write.req.bits.mask
        acc_addr_reg := io.mem_req.write.req.bits.addr
        state        := accReadResp
      }
    }
  }

  when(state === accReadResp && io.sramRead.resp.fire) {
    old_data_reg := io.sramRead.resp.bits.data
    state        := accWriteReq
  }

  when(state === accWriteReq && io.sramWrite.req.fire) {
    state := accWriteResp
  }

  when(state === accWriteResp && io.sramWrite.resp.valid) {
    state := idle
  }

  io.busy := state =/= idle
}
