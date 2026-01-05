package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * AccPipe: Accumulator Pipeline
 * Handles read-modify-write operations for accumulation
 * Separated from SramBank for flexibility
 */
@instantiable
class AccPipe(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    // Interface to SramBank - AccPipe sends requests to bank (sender perspective)
    val sramRead  = Flipped(new SramReadIO(b))
    val sramWrite = Flipped(new SramWriteIO(b))

    // Interface from midend - AccPipe receives requests (receiver perspective)
    val read  = new SramReadIO(b)
    val write = new SramWriteIO(b)

    // Control and status signals
    val bank_id         = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val current_bank_id = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val busy            = Output(Bool())
  })

  // Internal registers
  val current_bank_id = Reg(UInt(log2Up(b.memDomain.bankNum).W))
  val busy            = Reg(Bool())

  // Connect to outputs
  io.current_bank_id := current_bank_id
  io.busy            := busy

  // State machine for read-modify-write
  val s_idle :: s_read :: s_accumulate :: s_write :: Nil = Enum(4)
  val state                                              = RegInit(s_idle)

  // Update current_bank_id and busy signal
  when(state === s_idle) {
    when(io.write.req.valid) {
      current_bank_id := io.bank_id
      busy            := true.B
    }.elsewhen(io.read.req.valid) {
      current_bank_id := io.bank_id
      busy            := true.B
    }.otherwise {
      busy := false.B
    }
  }.otherwise {
    busy := true.B
  }

  // Pipeline registers
  val acc_addr = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  val acc_data = RegInit(0.U(b.memDomain.bankWidth.W))
  val acc_mask = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B)))

  // Default values for all outputs
  io.write.req.ready    := false.B
  io.write.resp.valid   := false.B
  io.write.resp.bits.ok := false.B

  io.read.req.ready      := false.B
  io.read.resp.valid     := false.B
  io.read.resp.bits.data := 0.U

  io.sramWrite.req.valid  := false.B
  io.sramWrite.req.bits   := 0.U.asTypeOf(io.sramWrite.req.bits)
  io.sramWrite.resp.ready := false.B

  io.sramRead.req.valid     := false.B
  io.sramRead.req.bits.addr := 0.U
  io.sramRead.resp.ready    := false.B

  io.current_bank_id := 0.U
  io.busy            := false.B

  // State machine logic
  when(state === s_idle) {
    when(io.write.req.valid) {
      // Direct write mode: pass through
      when(!io.write.req.bits.wmode) {
        io.sramWrite.req.valid     := io.write.req.valid
        io.sramWrite.req.bits.addr := io.write.req.bits.addr
        io.sramWrite.req.bits.data := io.write.req.bits.data
        io.sramWrite.req.bits.mask := io.write.req.bits.mask
        io.write.req.ready         := io.sramWrite.req.ready

        // Forward write response from bank
        io.write.resp.valid     := io.sramWrite.resp.valid
        io.write.resp.bits.ok   := io.sramWrite.resp.bits.ok
        io.sramWrite.resp.ready := io.write.resp.ready

        when(io.write.req.fire) {
          state := s_idle
          busy  := false.B
        }
      }.otherwise {
        // Accumulator mode: start read-modify-write
        io.sramRead.req.valid     := true.B
        io.sramRead.req.bits.addr := io.write.req.bits.addr
        io.write.req.ready        := io.sramRead.req.ready
        acc_addr                  := io.write.req.bits.addr
        acc_data                  := io.write.req.bits.data
        acc_mask                  := io.write.req.bits.mask
        state                     := s_read
      }
    }.elsewhen(io.read.req.valid) {
      // Pure read operation (not accumulation)
      io.sramRead.req.valid     := io.read.req.valid
      io.sramRead.req.bits.addr := io.read.req.bits.addr
      io.read.req.ready         := io.sramRead.req.ready
      state                     := s_read
    }
  }.elsewhen(state === s_read) {
    // Wait for read response
    when(io.sramRead.resp.valid) {
      // If we got here from a write request, it's accumulation
      // If we got here from a read request, it's a pure read
      when(io.read.req.valid) {
        // Pure read: pass data back immediately
        io.read.resp.valid     := io.sramRead.resp.valid
        io.read.resp.bits.data := io.sramRead.resp.bits.data
        io.sramRead.resp.ready := io.read.resp.ready

        when(io.read.resp.ready) {
          state := s_idle
          busy  := false.B
        }
      }.otherwise {
        // Accumulation: proceed to accumulate
        state := s_accumulate
      }
    }
  }.elsewhen(state === s_accumulate) {
    // Accumulate: old_data + new_data
    val new_data = io.sramRead.resp.bits.data + acc_data
    acc_data := new_data

    state := s_write
  }.elsewhen(state === s_write) {
    // Write back accumulated result
    io.sramWrite.req.valid     := true.B
    io.sramWrite.req.bits.addr := acc_addr
    io.sramWrite.req.bits.data := acc_data
    io.sramWrite.req.bits.mask := acc_mask

    // Forward write response from bank
    io.write.resp.valid     := io.sramWrite.resp.valid
    io.write.resp.bits.ok   := io.sramWrite.resp.bits.ok
    io.sramWrite.resp.ready := io.write.resp.ready

    when(io.sramWrite.req.fire) {
      state := s_idle
      busy  := false.B
    }
  }
}
