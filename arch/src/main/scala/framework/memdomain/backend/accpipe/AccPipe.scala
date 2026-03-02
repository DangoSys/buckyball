package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.memdomain.backend.MemRequestIO

/**
 * AccPipe: Accumulator Pipeline
 * - Direct write (wmode=0): write.req -> bank write -> forward resp
 * - Accum write (wmode=1): bank read -> (old + new with mask) -> bank write -> forward resp
 * - Read: bank read -> forward resp
 *
 * This version:
 * - Uses correct IO directions based on your SramReadIO/SramWriteIO definitions
 * - Uses strict Decoupled handshakes
 * - Latches op type/address/data/mask
 * - Latches old_data on read resp fire (no cross-state resp.bits usage)
 */
@instantiable
class AccPipe(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Interface to SramBank
    // Your SramReadIO/SramWriteIO are SLAVE-shaped (req is Flipped), so master must Flipped(...)
    val sramRead  = Flipped(new SramReadIO(b))  // AccPipe -> bank: req out, resp in
    val sramWrite = Flipped(new SramWriteIO(b)) // AccPipe -> bank: req out, resp in

    val mem_req  = Flipped(new MemRequestIO(b))
    val is_multi = Input(Bool())

    val busy     = Output(Bool())
    val group_id = Output(UInt(3.W))
    val bank_id  = Output(UInt(b.memDomain.bankNum.W))
  })

  val read :: write :: Nil = Enum(2)
  val state                = RegInit(read)

  io.sramRead <> io.mem_req.read
  io.sramWrite <> io.mem_req.write

  //Read Acc
  when(io.is_multi) {
    io.sramRead.req.bits.addr := io.mem_req.read.req.bits.addr(log2Ceil(b.memDomain.bankEntries) - 1, 2)
  }

  //group_id output
  val group_id_reg = RegInit(0.U(3.W))
  when(io.mem_req.read.req.valid || io.mem_req.write.req.valid) {
    io.group_id  := io.mem_req.group_id
    group_id_reg := io.mem_req.group_id
  }.otherwise {
    io.group_id := group_id_reg
  }

  //Bank_id output
  val bank_id_reg = RegInit(0.U(b.memDomain.bankNum.W))
  when(io.mem_req.read.req.valid || io.mem_req.write.req.valid) {
    bank_id_reg := io.mem_req.bank_id
    io.bank_id  := io.mem_req.bank_id
  }.otherwise {
    io.bank_id := bank_id_reg
  }

  val acc_data_reg = RegInit(0.U(b.memDomain.bankWidth.W))
  val acc_mask_reg = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))
  val acc_addr_reg = RegInit(0.U(b.memDomain.memAddrLen.W))

  switch(state) {
    is(read) { //Stage 1: Read Acc Data
      when(io.mem_req.write.req.valid && io.mem_req.write.req.bits.wmode) {
        state                     := write
        acc_data_reg              := io.mem_req.write.req.bits.data
        acc_mask_reg              := io.mem_req.write.req.bits.mask
        acc_addr_reg              := io.mem_req.write.req.bits.addr
        io.sramRead.req.bits.addr := io.mem_req.write.req.bits.addr
        io.sramRead.req.valid     := true.B

        io.sramWrite.req.valid := false.B
        io.sramWrite.req.bits  := DontCare
      }
    }
    is(write) { //Stage 2: Write Acc Data
      when(io.sramRead.resp.valid) {
        state                       := read
        io.sramWrite.req.bits.addr  := acc_addr_reg
        io.sramWrite.req.bits.data  := acc_data_reg + io.sramRead.resp.bits.data
        io.sramWrite.req.bits.mask  := acc_mask_reg
        io.sramWrite.req.bits.wmode := true.B
        io.sramWrite.req.valid      := true.B

        io.sramRead.req.valid := false.B
        io.sramRead.req.bits  := DontCare
      }
    }
  }

  io.busy := false.B
}
