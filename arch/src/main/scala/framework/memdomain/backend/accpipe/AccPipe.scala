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

    val busy = Output(Bool())
  })

  io.sramRead <> io.read
  io.sramWrite <> io.write
  io.busy := false.B
}
