package framework.builtin.memdomain.mem

import chisel3._
import chisel3.util._

import framework.builtin.util.Util._

class SramReadReq(val n: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val fromDMA = Bool()
}

class SramReadResp(val w: Int) extends Bundle {
  val data = UInt(w.W)
  val fromDMA = Bool()
}

class SramReadIO(val n: Int, val w: Int) extends Bundle {
  val req = Flipped(Decoupled(new SramReadReq(n)))
  val resp = Decoupled(new SramReadResp(w))
}

class SramWriteReq(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val addr = UInt(log2Ceil(n).W)
  val mask = Vec(mask_len, Bool())
  val data = UInt(w.W)
}

class SramWriteIO(val n: Int, val w: Int, val mask_len: Int) extends Bundle {
  val req = Flipped(Decoupled(new SramWriteReq(n, w, mask_len)))
}

class SramBank(n: Int, w: Int, aligned_to: Int, single_ported: Boolean) extends Module {
  require(w % aligned_to == 0 || w < aligned_to)

  val mask_len = (w / (aligned_to * 8)) max 1 // How many mask bits are there?
  val mask_elem = UInt((w min (aligned_to * 8)).W) // What datatype does each mask bit correspond to?

  val io = IO(new Bundle {
    val read = new SramReadIO(n, w)
    val write = new SramWriteIO(n, w, mask_len)
  })

  val mem = SyncReadMem(n, Vec(mask_len, mask_elem))

  // Note: Memory is not initialized on reset to avoid FIRRTL compilation explosion
  // Software should initialize memory before use if needed

  // Only one request per cycle is allowed
  assert(!(io.read.req.valid && io.write.req.valid), "SramBank: Read and write requests is not allowed at the same time")

  // Read request can be ready as long as there is no write request
  io.read.req.ready := !io.write.req.valid

// -----------------------------------------------------------------------------
// Write
// -----------------------------------------------------------------------------
  // Write request is always ready unless there is a read request in progress
  io.write.req.ready := !io.read.req.valid

  when (io.write.req.valid) {
      mem.write(io.write.req.bits.addr, io.write.req.bits.data.asTypeOf(Vec(mask_len, mask_elem)), VecInit((~(0.U(mask_len.W))).asBools))
  }

// -----------------------------------------------------------------------------
// Read
// -----------------------------------------------------------------------------
  val raddr   = io.read.req.bits.addr
  val ren     = io.read.req.fire
  val rdata   = mem.read(raddr, ren).asUInt
  val fromDMA = io.read.req.bits.fromDMA

  // Make a queue which buffers the result of an SRAM read if it can't immediately be consumed
  // val q = Module(new Queue(new SramReadResp(w), 1, true, true))
  // q.io.enq.valid        := RegNext(ren)
  // q.io.enq.bits.data    := rdata
  // q.io.enq.bits.fromDMA := RegNext(fromDMA)

  // io.read.resp <> q.io.deq
  io.read.resp.valid        := RegNext(ren)
  io.read.resp.bits.data    := rdata
  io.read.resp.bits.fromDMA := RegNext(fromDMA)
}
