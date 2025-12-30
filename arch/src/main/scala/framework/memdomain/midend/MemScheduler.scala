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

    // Input from frontend (Ball Domain read/write requests) - receiver perspective
    val frontend = new Bundle {
      val bankRead  = Vec(b.memDomain.bankNum, new BankRead(b))
      val bankWrite = Vec(b.memDomain.bankNum, new BankWrite(b))
    }

    // Output to backend (MemManager) - MemManager expects Flipped, so we don't flip here
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Basic direct connection: route frontend requests to backend channels
  // -----------------------------------------------------------------------------
  // Simple mapping: each channel handles requests for a specific bank
  // Channel ch handles bank (ch % bankNum)
  // Since bankChannel == bankNum (both 32), this is a 1:1 mapping

  for (ch <- 0 until b.memDomain.bankChannel) {
    val bankIdx = ch % b.memDomain.bankNum

    // Direct 1:1 connection
    io.mem_req(ch).write <> io.frontend.bankWrite(bankIdx).io
    io.mem_req(ch).read <> io.frontend.bankRead(bankIdx).io
    io.mem_req(ch).bank_id := io.frontend.bankWrite(bankIdx).bank_id
  }
}
