package memcore.bus.chi

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class RxIO(flitBits: Int) extends Bundle {
  val active = Input(Bool())
  val link   = Flipped(new Channel(flitBits))
  val out    = Decoupled(UInt(flitBits.W))
}

@instantiable
class Rx(flitBits: Int, depth: Int = 4) extends Module {
  require(flitBits >= 1)
  require(depth >= 1 && depth <= 15)

  @public
  val io = IO(new RxIO(flitBits))

  val pointerBits = math.max(1, log2Ceil(depth))
  val countBits   = log2Ceil(depth + 1)

  // Receive FIFO state
  val storage      = Reg(Vec(depth, UInt(flitBits.W)))
  val readPointer  = RegInit(0.U(pointerBits.W))
  val writePointer = RegInit(0.U(pointerBits.W))
  val occupancy    = RegInit(0.U(countBits.W))

  val dequeue      = occupancy =/= 0.U && io.out.ready
  val enqueueReady = occupancy < depth.U || dequeue
  val enqueue      = io.link.flitv && enqueueReady

  // Internal receive interface
  io.out.valid := occupancy =/= 0.U
  io.out.bits  := storage(readPointer)

  when(enqueue) {
    storage(writePointer) := io.link.flit
    writePointer          := Mux(writePointer === (depth - 1).U, 0.U, writePointer + 1.U)
  }
  when(dequeue) {
    readPointer := Mux(readPointer === (depth - 1).U, 0.U, readPointer + 1.U)
  }
  when(enqueue =/= dequeue) {
    occupancy := Mux(enqueue, occupancy + 1.U, occupancy - 1.U)
  }

  // Returned-credit state
  val advertised = RegInit(0.U(countBits.W))
  val grant      = io.active && (advertised +& occupancy) < depth.U

  io.link.lcrdv := grant
  when(grant =/= io.link.flitv) {
    advertised := Mux(grant, advertised + 1.U, advertised - 1.U)
  }

  // Link contract checks
  val previousPend = RegNext(io.link.flitpend, false.B)
  val wasActive    = RegNext(io.active, false.B)

  when(io.link.flitv) {
    assert(io.active, "CHI RX flit on inactive link")
    assert(previousPend, "CHI FLITPEND must precede FLITV")
    assert(advertised =/= 0.U, "CHI RX flit without granted credit")
    assert(enqueueReady, "CHI RX buffer overflow")
  }
  assert((advertised +& occupancy) <= depth.U, "CHI RX credit conservation")
  when(wasActive)(assert(io.active, "CHI RX requires coordinated reset to stop"))
}
