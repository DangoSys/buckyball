package framework.memdomain.midend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.MemRequestIO

/**
 * MemScheduler: Midend module for memory scheduling
 * Connects MemController to MemManager
 *
 * Basic direct connection: routes requests from frontend to backend channels
 */
@instantiable
class MemScheduler(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    // Input from frontend (MemController)
    val frontend = new Bundle {
      val bankRead  = Vec(b.memDomain.bankNum, Flipped(new BankRead(b)))
      val bankWrite = Vec(b.memDomain.bankNum, Flipped(new BankWrite(b)))
    }

    // Output to backend (MemManager)
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Basic direct connection: route frontend requests to backend channels
  // -----------------------------------------------------------------------------
  // Simple mapping: each channel handles requests for a specific bank
  // Channel ch handles bank (ch % bankNum)

  for (ch <- 0 until b.memDomain.bankChannel) {
    val targetBank = (ch % b.memDomain.bankNum).U
    val bankIdx    = ch  % b.memDomain.bankNum

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
