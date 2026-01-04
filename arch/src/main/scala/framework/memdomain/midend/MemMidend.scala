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
      val bankRead  = Vec(b.ballDomain.bbusConsumerChannels, new BankRead(b))
      val bankWrite = Vec(b.ballDomain.bbusProducerChannels, new BankWrite(b))
    }

    // Output to backend (MemManager) - MemManager expects Flipped, so we don't flip here
    val mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
  })

  // -----------------------------------------------------------------------------
  // Basic direct connection: route frontend requests to backend channels
  // -----------------------------------------------------------------------------
  // Map bbusChannel input channels to bankChannel output channels
  // Each input channel is mapped to a backend channel based on modulo

  for (ch <- 0 until b.memDomain.bankChannel) {
    val inputIdx = ch % b.ballDomain.bbusConsumerChannels

    // Map input channels to backend channels
    io.mem_req(ch).write <> io.frontend.bankWrite(inputIdx).io
    io.mem_req(ch).read <> io.frontend.bankRead(inputIdx).io
    io.mem_req(ch).bank_id := io.frontend.bankWrite(inputIdx).bank_id
  }
}
