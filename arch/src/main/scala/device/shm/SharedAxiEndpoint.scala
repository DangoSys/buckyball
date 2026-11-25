package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.bus.axi4
import framework.generic.queue.Queue

/** One read and one write burst, with four ordered SRAM operations in flight per direction. */
@instantiable
class SharedAxiEndpoint(b: Params) extends Module {
  val p = b

  @public val io = IO(new Bundle {
    val busy  = Output(Bool())
    val mem   = Flipped(new axi4.Port(p.axi))
    val read  = new StorageAccess(b.axi.dataBits)
    val write = new StorageAccess(b.axi.dataBits)
  })

  val live                              = !reset.asBool
  val reading                           = RegInit(false.B)
  val rAddress                          = Reg(UInt(p.sharedAddressBits.W))
  val rId                               = Reg(UInt(p.axi.idBits.W))
  val rBeats                            = Reg(UInt(9.W))
  val rIssued                           = RegInit(0.U(9.W))
  val rSent                             = RegInit(0.U(9.W))
  val data                              = Instantiate(new Queue(UInt(128.W), 4))
  val wIdle :: wStream :: wReply :: Nil = Enum(3)
  val writing                           = RegInit(wIdle)
  val wAddress                          = Reg(UInt(p.sharedAddressBits.W))
  val wId                               = Reg(UInt(p.axi.idBits.W))
  val wBeats                            = Reg(UInt(9.W))
  val wAccepted                         = RegInit(0.U(9.W))
  val wIssued                           = RegInit(0.U(9.W))
  val wCompleted                        = RegInit(0.U(9.W))
  val writes                            = Instantiate(new Queue(new StorageRequest(128), 4))
  io.busy := reading || writing =/= wIdle

  def check(address: axi4.Address): Unit = {
    val bytes = (address.len +& 1.U) << 4
    assert(
      address.size === 4.U && address.burst === 1.U && !address.lock && address.addr(3, 0) === 0.U,
      "C2C shared SRAM requires unlocked aligned 128-bit INCR bursts"
    )
    assert(address.addr +& bytes <= p.capacity.U, "C2C AXI address exceeds Hart shared SRAM")
    assert(address.addr.pad(12)(11, 0) +& bytes <= 4096.U, "C2C AXI burst crosses 4KiB")
  }

  io.mem.ar.ready                     := !reading && live
  when(io.mem.ar.fire) {
    check(io.mem.ar.bits)
    rAddress := io.mem.ar.bits.addr
    rId      := io.mem.ar.bits.id
    rBeats   := io.mem.ar.bits.len +& 1.U
    rIssued  := 0.U
    rSent    := 0.U
    reading  := true.B
  }
  io.read.request.valid               := reading && rIssued < rBeats && rIssued - rSent < 4.U && live
  io.read.request.bits.address        := rAddress + (rIssued << 4)
  io.read.request.bits.write          := false.B
  io.read.request.bits.data           := 0.U
  io.read.request.bits.mask           := 0.U
  when(io.read.request.fire)(rIssued  := rIssued + 1.U)
  data.io.enq.valid                   := reading && io.read.response.valid && live
  data.io.enq.bits                    := io.read.response.bits.data
  io.read.response.ready              := reading && data.io.enq.ready && live
  when(io.read.response.fire)(assert(!io.read.response.bits.error, "C2C shared SRAM read failed"))
  io.mem.r.valid                      := reading && data.io.deq.valid && live
  io.mem.r.bits.id                    := rId
  io.mem.r.bits.data                  := data.io.deq.bits
  io.mem.r.bits.resp                  := 0.U
  io.mem.r.bits.last                  := rSent +& 1.U === rBeats
  data.io.deq.ready                   := reading && io.mem.r.ready && live
  when(io.mem.r.fire) {
    rSent                                 := rSent + 1.U
    when(rSent +& 1.U === rBeats)(reading := false.B)
  }
  io.mem.aw.ready                     := writing === wIdle && live
  when(io.mem.aw.fire) {
    check(io.mem.aw.bits)
    wAddress   := io.mem.aw.bits.addr
    wId        := io.mem.aw.bits.id
    wBeats     := io.mem.aw.bits.len +& 1.U
    wAccepted  := 0.U
    wIssued    := 0.U
    wCompleted := 0.U
    writing    := wStream
  }
  writes.io.enq.valid                 := writing === wStream && wAccepted < wBeats && io.mem.w.valid && live
  writes.io.enq.bits.address          := wAddress + (wAccepted << 4)
  writes.io.enq.bits.write            := true.B
  writes.io.enq.bits.data             := io.mem.w.bits.data
  writes.io.enq.bits.mask             := io.mem.w.bits.strb
  io.mem.w.ready                      := writing === wStream && wAccepted < wBeats && writes.io.enq.ready && live
  when(io.mem.w.fire) {
    assert(io.mem.w.bits.last === (wAccepted +& 1.U === wBeats), "C2C AXI WLAST does not match AWLEN")
    wAccepted := wAccepted + 1.U
  }
  io.write.request.valid              := writing === wStream && writes.io.deq.valid && live
  io.write.request.bits               := writes.io.deq.bits
  writes.io.deq.ready                 := writing === wStream && io.write.request.ready && live
  when(io.write.request.fire)(wIssued := wIssued + 1.U)
  io.write.response.ready             := writing === wStream && live
  when(io.write.response.fire) {
    assert(wCompleted < wIssued && !io.write.response.bits.error, "C2C shared SRAM write failed")
    wCompleted                                 := wCompleted + 1.U
    when(wCompleted +& 1.U === wBeats)(writing := wReply)
  }
  io.mem.b.valid                      := writing === wReply && live
  io.mem.b.bits.id                    := wId
  io.mem.b.bits.resp                  := 0.U
  when(io.mem.b.fire)(writing         := wIdle)
}
