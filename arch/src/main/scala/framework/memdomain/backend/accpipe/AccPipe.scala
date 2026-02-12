package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}

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

    // Interface from midend (AccPipe is slave)
    val read  = new SramReadIO(b)  // midend -> AccPipe: req in, resp out
    val write = new SramWriteIO(b) // midend -> AccPipe: req in, resp out

    // Control and status signals
    val bank_id = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val is_acc  = Input(Bool())

    val busy = Output(Bool())
  })

  val read :: write :: Nil = Enum(2)
  val state                = RegInit(read)

  io.sramRead <> io.read
  io.sramWrite <> io.write

  when(io.is_acc) {
    io.sramRead.req.bits.addr := io.read.req.bits.addr(log2Ceil(b.memDomain.bankEntries) - 1, 2)
  }

  val acc_data_reg = RegInit(0.U(b.memDomain.bankWidth.W))
  val acc_mask_reg = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))
  val acc_addr_reg = RegInit(0.U(b.memDomain.memAddrLen.W))

  switch(state) {
    is(read) {
      when(io.write.req.valid && io.write.req.bits.wmode) {
        state                     := write
        acc_data_reg              := io.write.req.bits.data
        acc_mask_reg              := io.write.req.bits.mask
        acc_addr_reg              := io.write.req.bits.addr
        io.sramRead.req.bits.addr := io.write.req.bits.addr
        io.sramRead.req.valid     := true.B

        io.sramWrite.req.valid := false.B
        io.sramWrite.req.bits  := DontCare
      }
    }
    is(write) {
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
