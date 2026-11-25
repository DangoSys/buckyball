package framework.generic.memory.bank

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.queue.Queue

class BankRequest(p: BankSetParams) extends Bundle {
  val addr  = UInt(p.rowBits.W)
  val write = Bool()
  val data  = UInt(p.dataBits.W)
  val mask  = UInt(p.bytes.W)
  val tag   = UInt(p.tagBits.W)
}

class BankResponse(p: BankSetParams) extends Bundle {
  val data  = UInt(p.dataBits.W)
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

/** One physical SRAM bank. Cache and NoC policy remain outside this module. */
@instantiable
class Bank(p: BankSetParams) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new BankRequest(p)))
    val response = Decoupled(new BankResponse(p))
  })

  val memory    = SyncReadMem(p.entriesPerBank, Vec(p.bytes, UInt(8.W)))
  val responses = Instantiate(new Queue(new BankResponse(p), 4))
  val occupied  = RegInit(0.U(3.W))
  io.response <> responses.io.deq
  io.request.ready := occupied < 4.U || io.response.fire

  val read = memory.readWrite(
    io.request.bits.addr,
    io.request.bits.data.asTypeOf(Vec(p.bytes, UInt(8.W))),
    io.request.bits.mask.asBools,
    io.request.fire,
    io.request.bits.write
  )

  val completed = RegNext(io.request.fire, false.B)
  val tag       = RegEnable(io.request.bits.tag, io.request.fire)
  val written   = RegEnable(io.request.bits.write, io.request.fire)
  responses.io.enq.valid      := completed
  responses.io.enq.bits.data  := Mux(written, 0.U, read.asUInt)
  responses.io.enq.bits.tag   := tag
  responses.io.enq.bits.error := false.B
  occupied                    := occupied + io.request.fire - io.response.fire
  when(completed) {
    assert(responses.io.enq.ready, "Bank response credit exhausted")
  }
}
