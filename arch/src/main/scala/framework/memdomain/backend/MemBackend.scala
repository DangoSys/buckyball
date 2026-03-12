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

    // Shared path — exposed to tile level for multi-core sharing.
    // bankChannel ports: one per midend channel of this core.
    val shared_mem_req = Vec(b.memDomain.bankChannel, new MemRequestIO(b))
    val shared_config  = Decoupled(new MemConfigerIO(b))

    // Query interface: shared query goes out, private query handled internally.
    val shared_query_vbank_id    = Output(UInt(8.W))
    val shared_query_group_count = Input(UInt(4.W))

    // Original query interface from frontend
    val query_vbank_id    = Input(UInt(8.W))
    val query_is_shared   = Input(Bool())
    val query_group_count = Output(UInt(4.W))
  })

  // Keep the private backend datapath unchanged and isolate it in a dedicated module.
  val privateBackend: Instance[PrivateMemBackend] = Instantiate(new PrivateMemBackend(b))

  // Route config to the selected backend only.
  val cfgToShared = io.config.bits.is_shared
  privateBackend.io.config.valid := io.config.valid && !cfgToShared
  privateBackend.io.config.bits  := io.config.bits
  io.shared_config.valid         := io.config.valid && cfgToShared
  io.shared_config.bits          := io.config.bits
  io.config.ready                := Mux(cfgToShared, io.shared_config.ready, privateBackend.io.config.ready)

  // Query routing
  privateBackend.io.query_vbank_id := io.query_vbank_id
  io.shared_query_vbank_id         := io.query_vbank_id
  io.query_group_count             := Mux(io.query_is_shared, io.shared_query_group_count, privateBackend.io.query_group_count)

  // Track whether a vbank is currently allocated in shared backend.
  // Ball requests do not carry explicit shared/private info, so they are routed by this table.
  private val vbankIdxWidth      = log2Up(b.memDomain.bankNum)
  val privateAllocByVbank        = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(false.B)))
  val sharedAllocByVbank         = RegInit(VecInit(Seq.fill(b.memDomain.bankNum)(false.B)))
  val cfgVbankIdx                = io.config.bits.vbank_id(vbankIdxWidth - 1, 0)
  when(io.config.fire) {
    when(io.config.bits.alloc) {
      when(io.config.bits.is_shared) {
        sharedAllocByVbank(cfgVbankIdx) := true.B
      }.otherwise {
        privateAllocByVbank(cfgVbankIdx) := true.B
      }
    }.otherwise {
      when(io.config.bits.is_shared) {
        sharedAllocByVbank(cfgVbankIdx) := false.B
      }.otherwise {
        privateAllocByVbank(cfgVbankIdx) := false.B
      }
    }
  }

  // Per-channel request routing: is_shared=0 -> private, is_shared=1 -> shared IO.
  // Route selection is latched at request fire to keep response demux stable.
  val readPending      = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val writePending     = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val readRouteShared  = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))
  val writeRouteShared = RegInit(VecInit(Seq.fill(b.memDomain.bankChannel)(false.B)))

  // Shared IO defaults
  for (i <- 0 until b.memDomain.bankChannel) {
    io.shared_mem_req(i).bank_id   := 0.U
    io.shared_mem_req(i).group_id  := 0.U
    io.shared_mem_req(i).is_shared := false.B
    io.shared_mem_req(i).hart_id   := 0.U

    io.shared_mem_req(i).read.req.valid  := false.B
    io.shared_mem_req(i).read.req.bits   := DontCare
    io.shared_mem_req(i).read.resp.ready := false.B

    io.shared_mem_req(i).write.req.valid  := false.B
    io.shared_mem_req(i).write.req.bits   := DontCare
    io.shared_mem_req(i).write.resp.ready := false.B
  }

  for (i <- 0 until b.memDomain.bankChannel) {
    val isBallChannel      = i < b.top.memBallChannelNum
    val hasPrivateAlloc    = privateAllocByVbank(io.mem_req(i).bank_id)
    val hasSharedAlloc     = sharedAllocByVbank(io.mem_req(i).bank_id)
    val hasBallReq         = io.mem_req(i).read.req.valid || io.mem_req(i).write.req.valid
    when(isBallChannel.B && hasBallReq) {
      assert(
        !(hasPrivateAlloc && hasSharedAlloc),
        "MemBackend ambiguous Ball route: idx=%d has both private and shared allocations\n",
        io.mem_req(i).bank_id
      )
    }
    val ballRouteShared    = hasSharedAlloc && !hasPrivateAlloc
    val useSharedReq       = Mux(isBallChannel.B, ballRouteShared, io.mem_req(i).is_shared)
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
    privateBackend.io.mem_req(i).is_shared := useSharedReq
    privateBackend.io.mem_req(i).hart_id   := io.mem_req(i).hart_id
    io.shared_mem_req(i).bank_id           := io.mem_req(i).bank_id
    io.shared_mem_req(i).group_id          := io.mem_req(i).group_id
    io.shared_mem_req(i).is_shared         := useSharedReq
    io.shared_mem_req(i).hart_id           := io.mem_req(i).hart_id

    // Read request route
    privateBackend.io.mem_req(i).read.req.valid := io.mem_req(i).read.req.valid && !useSharedReq
    privateBackend.io.mem_req(i).read.req.bits  := io.mem_req(i).read.req.bits
    io.shared_mem_req(i).read.req.valid         := io.mem_req(i).read.req.valid && useSharedReq
    io.shared_mem_req(i).read.req.bits          := io.mem_req(i).read.req.bits
    io.mem_req(i).read.req.ready                := Mux(
      useSharedReq,
      io.shared_mem_req(i).read.req.ready,
      privateBackend.io.mem_req(i).read.req.ready
    )

    // Write request route
    privateBackend.io.mem_req(i).write.req.valid := io.mem_req(i).write.req.valid && !useSharedReq
    privateBackend.io.mem_req(i).write.req.bits  := io.mem_req(i).write.req.bits
    io.shared_mem_req(i).write.req.valid         := io.mem_req(i).write.req.valid && useSharedReq
    io.shared_mem_req(i).write.req.bits          := io.mem_req(i).write.req.bits
    io.mem_req(i).write.req.ready                := Mux(
      useSharedReq,
      io.shared_mem_req(i).write.req.ready,
      privateBackend.io.mem_req(i).write.req.ready
    )

    // Response ready route (selected by latched request route when pending).
    privateBackend.io.mem_req(i).read.resp.ready  := io.mem_req(i).read.resp.ready && !useSharedReadResp
    io.shared_mem_req(i).read.resp.ready          := io.mem_req(i).read.resp.ready && useSharedReadResp
    privateBackend.io.mem_req(i).write.resp.ready := io.mem_req(i).write.resp.ready && !useSharedWriteResp
    io.shared_mem_req(i).write.resp.ready         := io.mem_req(i).write.resp.ready && useSharedWriteResp

    // Response valid/bits mux back to midend.
    io.mem_req(i).read.resp.valid := Mux(
      useSharedReadResp,
      io.shared_mem_req(i).read.resp.valid,
      privateBackend.io.mem_req(i).read.resp.valid
    )
    io.mem_req(i).read.resp.bits  := Mux(
      useSharedReadResp,
      io.shared_mem_req(i).read.resp.bits,
      privateBackend.io.mem_req(i).read.resp.bits
    )

    io.mem_req(i).write.resp.valid := Mux(
      useSharedWriteResp,
      io.shared_mem_req(i).write.resp.valid,
      privateBackend.io.mem_req(i).write.resp.valid
    )
    io.mem_req(i).write.resp.bits  := Mux(
      useSharedWriteResp,
      io.shared_mem_req(i).write.resp.bits,
      privateBackend.io.mem_req(i).write.resp.bits
    )
  }
}
