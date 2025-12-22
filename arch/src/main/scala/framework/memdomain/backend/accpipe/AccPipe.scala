package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.MemDomainParam
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO, SramWriteReq, SramWriteResp}
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

/**
 * AccPipe: Accumulator Pipeline
 * Handles read-modify-write operations for accumulation
 * Separated from SramBank for flexibility
 */
@instantiable
class AccPipe(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {
    // Interface to SramBank
    val sramRead  = new SramReadIO(parameter.bankEntries, parameter.bankWidth)
    val sramWrite = new SramWriteIO(parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen)

    // Interface from midend
    val read  = Flipped(new SramReadIO(parameter.bankEntries, parameter.bankWidth))
    val write = Flipped(new SramWriteIO(parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen))

    // Control signals
    val bank_id = Input(UInt(log2Up(parameter.bankNum).W))
  })

  @public
  val current_bank_id = Reg(UInt(log2Up(parameter.bankNum).W))
  @public
  val busy            = Reg(Bool())

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
  val acc_addr = RegInit(0.U(log2Ceil(parameter.bankEntries).W))
  val acc_data = RegInit(0.U(parameter.bankWidth.W))
  val acc_mask = RegInit(VecInit(Seq.fill(parameter.bankMaskLen)(false.B)))

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

        io.write.resp.valid    := io.sramWrite.req.ready
        io.write.resp.bits.ok  := io.sramWrite.req.ready
        io.sramWrite.req.ready := io.write.resp.ready

        when(io.sramWrite.req.ready) {
          state := s_idle
          busy  := false.B
        }
      }.otherwise {
        // Accumulator mode: start read-modify-write
        io.sramRead.req.valid        := true.B
        io.sramRead.req.bits.addr    := io.write.req.bits.addr
        io.sramRead.req.bits.fromDMA := false.B
        io.write.req.ready           := io.sramRead.req.ready

        acc_addr := io.write.req.bits.addr
        acc_data := io.write.req.bits.data
        acc_mask := io.write.req.bits.mask

        state := s_read
      }
    }.elsewhen(io.read.req.valid) {
      // Pure read operation (not accumulation)
      io.sramRead.req.valid        := io.read.req.valid
      io.sramRead.req.bits.addr    := io.read.req.bits.addr
      io.sramRead.req.bits.fromDMA := false.B
      io.read.req.ready            := io.sramRead.req.ready

      state := s_read
    }
  }.elsewhen(state === s_read) {
    // Wait for read response
    when(io.sramRead.resp.valid) {
      // If we got here from a write request, it's accumulation
      // If we got here from a read request, it's a pure read
      when(io.read.req.valid) {
        // Pure read: pass data back immediately
        io.read.resp.valid        := io.sramRead.resp.valid
        io.read.resp.bits.data    := io.sramRead.resp.bits.data
        io.read.resp.bits.fromDMA := io.sramRead.resp.bits.fromDMA
        io.sramRead.resp.ready    := io.read.resp.ready

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

    io.write.resp.valid    := io.sramWrite.req.ready
    io.write.resp.bits.ok  := io.sramWrite.req.ready
    io.sramWrite.req.ready := io.write.resp.ready

    when(io.sramWrite.req.ready) {
      state := s_idle
      busy  := false.B
    }
  }

  // Tie off unused signals in unused states
  when(state === s_read) {
    io.sramRead.resp.ready := true.B
  }.otherwise {
    io.sramRead.resp.ready := false.B
  }

  when(state =/= s_write && state =/= s_idle) {
    io.sramWrite.req.valid := false.B
  }
}
