package framework.memdomain.midend

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.memdomain.backend.MemRequestIO

/**
 * MemMidend: Midend module for memory scheduling
 * Connects MemFrontend to MemManager
 *
 * Basic direct connection: routes requests from frontend to backend channels
 */
@instantiable
class MemMidend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    // Input from frontend (Ball Domain read/write requests) - receiver perspective
    val frontend = new Bundle {
      val bankRead  = new BankRead(b)
      val bankWrite = new BankWrite(b)
    }

    val balldomain = new Bundle {
      val bankRead  = Vec(b.top.memBallChannelNum, new BankRead(b))
      val bankWrite = Vec(b.top.ballMemChannelNum, new BankWrite(b))
    }

    // Output to backend (MemManager) - MemManager expects Flipped, so we don't flip here
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Basic direct connection: route frontend requests to backend channels
  // -----------------------------------------------------------------------------
  // Map bbusChannel input channels to bankChannel output channels
  // Each input channel is mapped to a backend channel based on modulo

  for (ch <- 0 until b.top.memBallChannelNum) {

    // Map input channels to backend channels
    io.mem_req(ch).write <> io.balldomain.bankWrite(ch).io
    io.mem_req(ch).read <> io.balldomain.bankRead(ch).io
    io.mem_req(ch).rbank_id := io.balldomain.bankWrite(ch).bank_id
    io.mem_req(ch).wbank_id := io.balldomain.bankWrite(ch).bank_id
  }
  io.mem_req(b.top.memBallChannelNum).write <> io.frontend.bankWrite.io
  io.mem_req(b.top.memBallChannelNum).read <> io.frontend.bankRead.io
  io.mem_req(b.top.memBallChannelNum).rbank_id := io.frontend.bankWrite.bank_id
  io.mem_req(b.top.memBallChannelNum).wbank_id := io.frontend.bankRead.bank_id
}
