package framework.memdomain.midend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import framework.memdomain.MemDomainParam
import framework.memdomain.backend.MemRequestIO

/**
 * MemScheduler: Midend module for memory scheduling
 * Connects MemController to MemManager
 *
 * Basic direct connection: routes requests from frontend to backend channels
 */
@instantiable
class MemScheduler(val parameter: MemDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[MemDomainParam] {

  @public
  val io = IO(new Bundle {

    // Input from frontend (MemController)
    val frontend = new Bundle {

      val bankRead = Vec(
        parameter.bankNum,
        Flipped(new BankRead(parameter.bankEntries, parameter.bankWidth, parameter.rob_entries, parameter.bankNum))
      )

      val bankWrite = Vec(
        parameter.bankNum,
        Flipped(new BankWrite(
          parameter.bankEntries,
          parameter.bankWidth,
          parameter.bankMaskLen,
          parameter.rob_entries,
          parameter.bankNum
        ))
      )

    }

    // Output to backend (MemManager)
    val mem_req = Vec(parameter.bankChannel, new MemRequestIO(parameter))
  })

  // -----------------------------------------------------------------------------
  // Basic direct connection: route frontend requests to backend channels
  // -----------------------------------------------------------------------------
  // Simple mapping: each channel handles requests for a specific bank
  // Channel ch handles bank (ch % bankNum)

  for (ch <- 0 until parameter.bankChannel) {
    val targetBank = (ch % parameter.bankNum).U
    val bankIdx    = ch  % parameter.bankNum

    // Default: no request
    io.mem_req(ch).write.req.valid := false.B
    io.mem_req(ch).read.req.valid  := false.B
    io.mem_req(ch).bank_id         := targetBank

    // Connect write request
    val writeValid = io.frontend.bankWrite(bankIdx).io.req.valid &&
      (io.frontend.bankWrite(bankIdx).bank_id === targetBank)
    when(writeValid) {
      io.mem_req(ch).write.req.valid               := io.frontend.bankWrite(bankIdx).io.req.valid
      io.mem_req(ch).write.req.bits                := io.frontend.bankWrite(bankIdx).io.req.bits
      io.frontend.bankWrite(bankIdx).io.req.ready  := io.mem_req(ch).write.req.ready
      io.frontend.bankWrite(bankIdx).io.resp.valid := io.mem_req(ch).write.resp.valid
      io.frontend.bankWrite(bankIdx).io.resp.bits  := io.mem_req(ch).write.resp.bits
      io.mem_req(ch).write.resp.ready              := io.frontend.bankWrite(bankIdx).io.resp.ready
      io.mem_req(ch).write.req.bits.wmode          := io.frontend.bankWrite(bankIdx).io.req.bits.wmode
      io.mem_req(ch).bank_id                       := io.frontend.bankWrite(bankIdx).bank_id
    }.otherwise {
      io.frontend.bankWrite(bankIdx).io.req.ready  := false.B
      io.frontend.bankWrite(bankIdx).io.resp.valid := false.B
      io.frontend.bankWrite(bankIdx).io.resp.ready := true.B
    }

    // Connect read request
    val readValid = io.frontend.bankRead(bankIdx).io.req.valid &&
      (io.frontend.bankRead(bankIdx).bank_id === targetBank)
    when(readValid) {
      io.mem_req(ch).read.req.valid               := io.frontend.bankRead(bankIdx).io.req.valid
      io.mem_req(ch).read.req.bits                := io.frontend.bankRead(bankIdx).io.req.bits
      io.frontend.bankRead(bankIdx).io.req.ready  := io.mem_req(ch).read.req.ready
      io.frontend.bankRead(bankIdx).io.resp.valid := io.mem_req(ch).read.resp.valid
      io.frontend.bankRead(bankIdx).io.resp.bits  := io.mem_req(ch).read.resp.bits
      io.mem_req(ch).read.resp.ready              := io.frontend.bankRead(bankIdx).io.resp.ready
      io.mem_req(ch).bank_id                      := io.frontend.bankRead(bankIdx).bank_id
    }.otherwise {
      io.frontend.bankRead(bankIdx).io.req.ready  := false.B
      io.frontend.bankRead(bankIdx).io.resp.valid := false.B
      io.frontend.bankRead(bankIdx).io.resp.ready := true.B
    }
  }
}
