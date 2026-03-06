package framework.memdomain.backend

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig
import framework.memdomain.backend.privatepath.PrivateMemBackend

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(8.W))
    val query_group_count = Output(UInt(4.W))
  })

  // Keep the private backend datapath unchanged and isolate it in a dedicated module.
  val privateBackend: Instance[PrivateMemBackend] = Instantiate(new PrivateMemBackend(b))

  privateBackend.io.mem_req <> io.mem_req
  privateBackend.io.config <> io.config
  privateBackend.io.query_vbank_id := io.query_vbank_id
  io.query_group_count             := privateBackend.io.query_group_count

  // Shared backend is intentionally not connected at top-level in this refactor
  // to avoid private-path regressions. It can be integrated in a later step.
}
