package framework.memdomain.backend

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import framework.memdomain.frontend.outside_channel.MemConfigerIO
import framework.top.GlobalConfig
import framework.memdomain.backend.privatepath.PrivateMemBackend
import framework.memdomain.backend.shared.SharedMemBackend

@instantiable
class MemBackend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val mem_req = Vec(b.memDomain.bankChannel, Flipped(new MemRequestIO(b)))
    val config  = Flipped(Decoupled(new MemConfigerIO(b)))

    // Query interface for frontend to get group count
    val query_vbank_id    = Input(UInt(8.W))
    val query_group_count = Output(UInt(4.W))

    val hartid = Input(UInt(b.core.xLen.W))
  })

  // Keep the private backend datapath unchanged and isolate it in a dedicated module.
  val privateBackend: Instance[PrivateMemBackend] = Instantiate(new PrivateMemBackend(b))
  val sharedBackend:  Instance[SharedMemBackend]  = Instantiate(new SharedMemBackend(b))

  // Broadcast config so alloc/release is handled by the corresponding backend.
  privateBackend.io.config.valid := io.config.valid
  privateBackend.io.config.bits  := io.config.bits
  sharedBackend.io.config.valid  := io.config.valid
  sharedBackend.io.config.bits   := io.config.bits
  io.config.ready                := privateBackend.io.config.ready && sharedBackend.io.config.ready

  // Query both backends; shared takes priority when this vbank is allocated as shared.
  privateBackend.io.query_vbank_id := io.query_vbank_id
  sharedBackend.io.query_vbank_id  := io.query_vbank_id
  sharedBackend.io.hartid          := io.hartid
  io.query_group_count             := Mux(
    sharedBackend.io.query_group_count =/= 0.U,
    sharedBackend.io.query_group_count,
    privateBackend.io.query_group_count
  )

  // Per-channel request routing: is_shared=0 -> private, is_shared=1 -> shared.
  // Route selection is latched at request fire to keep response demux stable.
  val readPending      = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val writePending     = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val readRouteShared  = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val writeRouteShared = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))

  for (i <- 0 until b.memDomain.bankChannel) {
    val useSharedReq       = io.mem_req(i).is_shared
    val useSharedReadResp  = Mux(readPending(i), readRouteShared(i), useSharedReq)
    val useSharedWriteResp = Mux(writePending(i), writeRouteShared(i), useSharedReq)

    when(io.mem_req(i).read.req.fire) {
      readPending(i)     := true.B
      readRouteShared(i) := useSharedReq
    }
    when(io.mem_req(i).read.resp.fire) {
      readPending(i) := false.B
    }

    when(io.mem_req(i).write.req.fire) {
      writePending(i)     := true.B
      writeRouteShared(i) := useSharedReq
    }
    when(io.mem_req(i).write.resp.fire) {
      writePending(i) := false.B
    }

    // Metadata is passed to both backends; only selected backend receives valid req/resp-ready.
    privateBackend.io.mem_req(i).bank_id   := io.mem_req(i).bank_id
    privateBackend.io.mem_req(i).group_id  := io.mem_req(i).group_id
    privateBackend.io.mem_req(i).is_shared := io.mem_req(i).is_shared
    sharedBackend.io.mem_req(i).bank_id    := io.mem_req(i).bank_id
    sharedBackend.io.mem_req(i).group_id   := io.mem_req(i).group_id
    sharedBackend.io.mem_req(i).is_shared  := io.mem_req(i).is_shared

    // Read request route
    privateBackend.io.mem_req(i).read.req.valid := io.mem_req(i).read.req.valid && !useSharedReq
    privateBackend.io.mem_req(i).read.req.bits  := io.mem_req(i).read.req.bits
    sharedBackend.io.mem_req(i).read.req.valid  := io.mem_req(i).read.req.valid && useSharedReq
    sharedBackend.io.mem_req(i).read.req.bits   := io.mem_req(i).read.req.bits
    io.mem_req(i).read.req.ready                := Mux(
      useSharedReq,
      sharedBackend.io.mem_req(i).read.req.ready,
      privateBackend.io.mem_req(i).read.req.ready
    )

    // Write request route
    privateBackend.io.mem_req(i).write.req.valid := io.mem_req(i).write.req.valid && !useSharedReq
    privateBackend.io.mem_req(i).write.req.bits  := io.mem_req(i).write.req.bits
    sharedBackend.io.mem_req(i).write.req.valid  := io.mem_req(i).write.req.valid && useSharedReq
    sharedBackend.io.mem_req(i).write.req.bits   := io.mem_req(i).write.req.bits
    io.mem_req(i).write.req.ready                := Mux(
      useSharedReq,
      sharedBackend.io.mem_req(i).write.req.ready,
      privateBackend.io.mem_req(i).write.req.ready
    )

    // Response ready route (selected by latched request route when pending).
    privateBackend.io.mem_req(i).read.resp.ready  := io.mem_req(i).read.resp.ready && !useSharedReadResp
    sharedBackend.io.mem_req(i).read.resp.ready   := io.mem_req(i).read.resp.ready && useSharedReadResp
    privateBackend.io.mem_req(i).write.resp.ready := io.mem_req(i).write.resp.ready && !useSharedWriteResp
    sharedBackend.io.mem_req(i).write.resp.ready  := io.mem_req(i).write.resp.ready && useSharedWriteResp

    // Response valid/bits mux back to midend.
    io.mem_req(i).read.resp.valid := Mux(
      useSharedReadResp,
      sharedBackend.io.mem_req(i).read.resp.valid,
      privateBackend.io.mem_req(i).read.resp.valid
    )
    io.mem_req(i).read.resp.bits  := Mux(
      useSharedReadResp,
      sharedBackend.io.mem_req(i).read.resp.bits,
      privateBackend.io.mem_req(i).read.resp.bits
    )

    io.mem_req(i).write.resp.valid := Mux(
      useSharedWriteResp,
      sharedBackend.io.mem_req(i).write.resp.valid,
      privateBackend.io.mem_req(i).write.resp.valid
    )
    io.mem_req(i).write.resp.bits  := Mux(
      useSharedWriteResp,
      sharedBackend.io.mem_req(i).write.resp.bits,
      privateBackend.io.mem_req(i).write.resp.bits
    )
  }
}
