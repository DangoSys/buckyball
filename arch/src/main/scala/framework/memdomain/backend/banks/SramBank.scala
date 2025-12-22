package framework.memdomain.backend.banks

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.MemDomainParam
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}

/**
 * SramBank: Pure SRAM bank
 * Simple read/write memory without any accumulation logic
 * Each bank is a single-port SRAM
 */
@instantiable
class SramBank(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {
  val aligned_to = 8
  val mask_len   = (parameter.bankWidth / (aligned_to * 8)) max 1
  val mask_elem  = UInt((parameter.bankWidth min (aligned_to * 8)).W)

  @public
  val io = IO(new Bundle {
    // Bank interface (connected to MemManager)
    val sramRead  = new SramReadIO(parameter.bankEntries, parameter.bankWidth)
    val sramWrite = new SramWriteIO(parameter.bankEntries, parameter.bankWidth, parameter.bankMaskLen)
  })

  // SRAM memory
  val mem = SyncReadMem(parameter.bankEntries, Vec(mask_len, mask_elem))

  // -----------------------------------------------------------------------------
  // Read path
  // -----------------------------------------------------------------------------
  // Bank read (single port - can't read and write simultaneously)
  io.sramRead.req.ready := !io.sramWrite.req.valid

  val raddr = io.sramRead.req.bits.addr
  val ren   = io.sramRead.req.fire
  val rdata = mem.read(raddr, ren)

  io.sramRead.resp.valid        := RegNext(ren)
  io.sramRead.resp.bits.data    := RegNext(rdata.asUInt)
  io.sramRead.resp.bits.fromDMA := false.B

  // -----------------------------------------------------------------------------
  // Write path
  // -----------------------------------------------------------------------------
  io.sramWrite.req.ready := !io.sramRead.req.valid

  when(io.sramWrite.req.valid) {
    mem.write(
      io.sramWrite.req.bits.addr,
      io.sramWrite.req.bits.data.asTypeOf(Vec(mask_len, mask_elem)),
      io.sramWrite.req.bits.mask
    )
  }
}
