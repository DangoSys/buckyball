package framework.acc.memdomain.bank_router.blt

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.ip.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class AccPipe(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Module {

  @public
  val io = IO(new Bundle {
    val sramRead  = Flipped(new SramReadIO(memParam))
    val sramWrite = Flipped(new SramWriteIO(memParam))

    val mem_req = new Bundle {
      val read      = new SramReadIO(memParam)
      val write     = new SramWriteIO(memParam)
      val bank_id   = Input(UInt(frontendParam.bank_id_len.W))
      val group_id  = Input(UInt(memParam.groupIdWidth.W))
      val thread_id = Input(UInt(64.W))
    }

    val busy      = Output(Bool())
    val group_id  = Output(UInt(memParam.groupIdWidth.W))
    val bank_id   = Output(UInt(frontendParam.bank_id_len.W))
    val thread_id = Output(UInt(64.W))
  })

  // Each group has its own physical bank, so no address shifting is needed.

  //group_id output
  val group_id_reg = RegInit(0.U(memParam.groupIdWidth.W))
  io.group_id := group_id_reg

  //Bank_id output
  val bank_id_reg = RegInit(0.U(frontendParam.bank_id_len.W))
  io.bank_id := bank_id_reg
  val thread_id_reg = RegInit(0.U(64.W))
  io.thread_id := thread_id_reg

  val rd_hold     = RegInit(false.B)
  val rd_data_reg = RegInit(0.U(memParam.bankWidth.W))
  val wr_hold     = RegInit(false.B)
  val wr_ok_reg   = RegInit(false.B)
  val busy_reg    = RegInit(false.B)

  val canStart    = !busy_reg && !rd_hold && !wr_hold
  val hasWriteReq = io.mem_req.write.req.valid
  val wrReq       = canStart && hasWriteReq
  val rdReq       = canStart && !hasWriteReq && io.mem_req.read.req.valid

  io.sramRead.req.valid     := rdReq
  io.sramRead.req.bits.addr := io.mem_req.read.req.bits.addr
  io.sramRead.resp.ready    := !rd_hold

  io.sramWrite.req.valid     := wrReq
  io.sramWrite.req.bits.addr := io.mem_req.write.req.bits.addr
  io.sramWrite.req.bits.data := io.mem_req.write.req.bits.data
  io.sramWrite.req.bits.mask := io.mem_req.write.req.bits.mask
  io.sramWrite.resp.ready    := !wr_hold

  io.mem_req.read.req.ready      := canStart && !hasWriteReq && io.sramRead.req.ready
  io.mem_req.read.resp.valid     := rd_hold
  io.mem_req.read.resp.bits.data := rd_data_reg

  io.mem_req.write.req.ready    := canStart && io.sramWrite.req.ready
  io.mem_req.write.resp.valid   := wr_hold
  io.mem_req.write.resp.bits.ok := wr_ok_reg

  when(rd_hold && io.mem_req.read.resp.ready) {
    rd_hold := false.B
  }.elsewhen(io.sramRead.resp.fire) {
    rd_hold     := true.B
    rd_data_reg := io.sramRead.resp.bits.data
  }

  when(wr_hold && io.mem_req.write.resp.ready) {
    wr_hold := false.B
  }.elsewhen(io.sramWrite.resp.fire) {
    wr_hold   := true.B
    wr_ok_reg := io.sramWrite.resp.bits.ok
  }

  when(io.mem_req.read.req.fire || io.mem_req.write.req.fire) {
    group_id_reg  := io.mem_req.group_id
    bank_id_reg   := io.mem_req.bank_id
    thread_id_reg := io.mem_req.thread_id
  }
  when(io.mem_req.read.req.fire || io.mem_req.write.req.fire) {
    busy_reg := true.B
  }
  when(io.mem_req.read.resp.fire || io.mem_req.write.resp.fire) {
    busy_reg := false.B
  }

  io.busy := busy_reg
}
