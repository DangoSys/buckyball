package framework.memdomain.backend.accpipe

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.MemDomainParam
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

/**
 * BankPipe: Bank Pipeline (直通，非流水线)
 * Direct connection to SramBank for read and write operations
 * Used for direct write/overwrite operations (non-accumulation)
 * Can also serve as a bypass when AccPipe is not fully utilized
 */
@instantiable
class BankPipe(val parameter: MemDomainParam)(implicit p: Parameters)
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

  // Update current_bank_id and busy signal
  when(io.write.req.valid || io.read.req.valid) {
    current_bank_id := io.bank_id
    busy            := true.B
  }.otherwise {
    busy := false.B
  }

  // -----------------------------------------------------------------------------
  // Write path (直通)
  // -----------------------------------------------------------------------------
  // Direct write: pass through to SramBank
  io.sramWrite.req <> io.write.req
  io.write.resp <> io.sramWrite.resp

  // -----------------------------------------------------------------------------
  // Read path (直通)
  // -----------------------------------------------------------------------------
  // Direct read: pass through to SramBank
  io.sramRead.req <> io.read.req
  io.read.resp <> io.sramRead.resp
}
