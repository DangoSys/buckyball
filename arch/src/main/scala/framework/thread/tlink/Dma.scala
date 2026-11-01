package framework.thread.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.bus.{axi, axi4}
import framework.lib.queue.Queue

class Address(p: axi4.Params) extends Bundle {
  val physical       = UInt(p.addressBits.W)
  val virtualAddress = UInt(64.W)
  val beats          = UInt(9.W)
  val id             = UInt(p.idBits.W)
}

class Completion(p: axi4.Params) extends Bundle {
  val id    = UInt(p.idBits.W)
  val error = Bool()
}

/** Credits account for link buffering, not memory completion. */
class Credit[T <: Data](payload: T) extends Bundle {
  val valid  = Output(Bool())
  val bits   = Output(payload.cloneType)
  val credit = Input(Bool())
}

/** The thread's ordinary DMA interface. Storage selection belongs to its core. */
class Dma(p: axi4.Params) extends Bundle {
  val read      = Decoupled(new Address(p))
  val write     = Decoupled(new Address(p))
  val writeData = new Credit(new axi.Beat(p.dataBits))
  val readData  = Flipped(new Credit(new axi.Beat(p.dataBits, p.idBits, 0, 2)))
  val completed = Flipped(Decoupled(new Completion(p)))
}

@instantiable
class Send[T <: Data](payload: T) extends Module {

  @public val io = IO(new Bundle {
    val input  = Flipped(Decoupled(payload.cloneType))
    val output = new Credit(payload)
  })

  val credits = RegInit(4.U(3.W))
  io.input.ready  := credits =/= 0.U && !reset.asBool
  io.output.valid := io.input.fire
  io.output.bits  := io.input.bits
  when(io.output.credit && !io.input.fire) {
    assert(credits < 4.U, "Thread DMA returned an unowned credit")
    credits := credits + 1.U
  }.elsewhen(io.input.fire && !io.output.credit) {
    credits := credits - 1.U
  }
}

@instantiable
class Receive[T <: Data](payload: T) extends Module {

  @public val io = IO(new Bundle {
    val input  = Flipped(new Credit(payload))
    val output = Decoupled(payload.cloneType)
  })

  val buffer = Instantiate(new Queue(payload.cloneType, 4, false, false))
  buffer.io.enq.valid := io.input.valid && !reset.asBool
  buffer.io.enq.bits  := io.input.bits
  io.output <> buffer.io.deq
  io.input.credit     := io.output.fire
  when(io.input.valid && !reset.asBool) {
    assert(buffer.io.enq.ready, "Thread DMA exceeded the credited capacity")
  }
}

/** Adapts the accelerator's DMA implementation to the thread boundary. */
@instantiable
class Stream(p: axi4.Params) extends Module {

  @public val io = IO(new Bundle {
    val acc          = Flipped(new axi4.Port(p))
    val readAddress  = Input(UInt(64.W))
    val writeAddress = Input(UInt(64.W))
    val dma          = new Dma(p)
  })

  io.dma.read.valid                := io.acc.ar.valid
  io.dma.read.bits.physical        := io.acc.ar.bits.addr
  io.dma.read.bits.virtualAddress  := io.readAddress
  io.dma.read.bits.beats           := io.acc.ar.bits.len +& 1.U
  io.dma.read.bits.id              := io.acc.ar.bits.id
  io.acc.ar.ready                  := io.dma.read.ready
  io.dma.write.valid               := io.acc.aw.valid
  io.dma.write.bits.physical       := io.acc.aw.bits.addr
  io.dma.write.bits.virtualAddress := io.writeAddress
  io.dma.write.bits.beats          := io.acc.aw.bits.len +& 1.U
  io.dma.write.bits.id             := io.acc.aw.bits.id
  io.acc.aw.ready                  := io.dma.write.ready
  for (address <- Seq(io.acc.ar, io.acc.aw)) {
    when(address.fire) {
      assert(
        address.bits.size === log2Ceil(p.bytes).U && address.bits.burst === 1.U && !address.bits.lock,
        "Thread DMA requires full-width incremental accesses"
      )
    }
  }
  val write = Instantiate(new Send(new axi.Beat(p.dataBits)))
  write.io.input.valid     := io.acc.w.valid
  write.io.input.bits      := 0.U.asTypeOf(write.io.input.bits)
  write.io.input.bits.data := io.acc.w.bits.data
  write.io.input.bits.keep := io.acc.w.bits.strb
  write.io.input.bits.last := io.acc.w.bits.last
  io.acc.w.ready           := write.io.input.ready
  io.dma.writeData <> write.io.output
  val read = Instantiate(new Receive(new axi.Beat(p.dataBits, p.idBits, 0, 2)))
  read.io.input <> io.dma.readData
  io.acc.r.valid         := read.io.output.valid
  io.acc.r.bits.id       := read.io.output.bits.id
  io.acc.r.bits.data     := read.io.output.bits.data
  io.acc.r.bits.resp     := read.io.output.bits.user
  io.acc.r.bits.last     := read.io.output.bits.last
  read.io.output.ready   := io.acc.r.ready
  io.acc.b.valid         := io.dma.completed.valid
  io.acc.b.bits.id       := io.dma.completed.bits.id
  io.acc.b.bits.resp     := Mux(io.dma.completed.bits.error, 2.U, 0.U)
  io.dma.completed.ready := io.acc.b.ready
}
