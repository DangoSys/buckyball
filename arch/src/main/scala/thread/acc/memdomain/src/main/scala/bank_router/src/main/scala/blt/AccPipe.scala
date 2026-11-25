package framework.thread.acc.memdomain.bank_router.blt

import framework.generic.memory.bank.{SramReadIO, SramReadReq, SramReadResp, SramWriteIO, SramWriteReq, SramWriteResp}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.memdomain.configs.MemDomainParam

@instantiable
class AccPipe(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Module {

  @public
  val io = IO(new Bundle {
    val sramRead  = Flipped(new SramReadIO(memParam.bankWidth))
    val sramWrite = Flipped(new SramWriteIO(memParam.bankWidth, memParam.bankMaskLen))

    val mem_req = new Bundle {
      val read      = new SramReadIO(memParam.bankWidth)
      val write     = new SramWriteIO(memParam.bankWidth, memParam.bankMaskLen)
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

  // Reads and writes are valid/ready pipelines: the SRAM output stage, then one register.
  val rd_valid    = RegInit(false.B)
  val rd_data_reg = RegInit(0.U(memParam.bankWidth.W))
  val wr_valid    = RegInit(false.B)
  val wr_ok_reg   = RegInit(false.B)

  val rd_busy     = rd_valid || io.sramRead.resp.valid
  val wr_busy     = wr_valid || io.sramWrite.resp.valid
  val hasWriteReq = io.mem_req.write.req.valid
  val wrReq       = !rd_busy && hasWriteReq
  val rdReq       = !wr_busy && !hasWriteReq && io.mem_req.read.req.valid

  io.sramRead.req.valid     := rdReq
  io.sramRead.req.bits.addr := io.mem_req.read.req.bits.addr
  io.sramRead.resp.ready    := !rd_valid || io.mem_req.read.resp.ready

  io.sramWrite.req.valid     := wrReq
  io.sramWrite.req.bits.addr := io.mem_req.write.req.bits.addr
  io.sramWrite.req.bits.data := io.mem_req.write.req.bits.data
  io.sramWrite.req.bits.mask := io.mem_req.write.req.bits.mask
  io.sramWrite.resp.ready    := !wr_valid || io.mem_req.write.resp.ready

  io.mem_req.read.req.ready      := !wr_busy && !hasWriteReq && io.sramRead.req.ready
  io.mem_req.read.resp.valid     := rd_valid
  io.mem_req.read.resp.bits.data := rd_data_reg

  io.mem_req.write.req.ready    := !rd_busy && io.sramWrite.req.ready
  io.mem_req.write.resp.valid   := wr_valid
  io.mem_req.write.resp.bits.ok := wr_ok_reg

  when(io.sramRead.resp.ready) {
    rd_valid := io.sramRead.resp.valid
  }
  when(io.sramRead.resp.fire) {
    rd_data_reg := io.sramRead.resp.bits.data
  }

  when(io.sramWrite.resp.ready) {
    wr_valid := io.sramWrite.resp.valid
  }
  when(io.sramWrite.resp.fire) {
    wr_ok_reg := io.sramWrite.resp.bits.ok
  }

  when(io.mem_req.read.req.fire || io.mem_req.write.req.fire) {
    group_id_reg  := io.mem_req.group_id
    bank_id_reg   := io.mem_req.bank_id
    thread_id_reg := io.mem_req.thread_id
  }
  io.busy := rd_busy || wr_busy
}
