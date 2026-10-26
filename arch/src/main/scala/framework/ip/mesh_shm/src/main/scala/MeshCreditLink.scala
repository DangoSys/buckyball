package framework.ip.memory.mesh_shm

import framework.lib.queue.Queue
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class MeshCreditPort(p: MeshSharedMemParams) extends Bundle {
  val valid  = Output(Bool())
  val bits   = Output(new MeshPacket(p))
  val credit = Input(Bool())
}

@instantiable
class MeshDecoupledToCredit(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshPacket(p)))
    val out = new MeshCreditPort(p)
  })

  val credits   = RegInit(p.linkBufferDepth.U(p.linkCreditBits.W))
  val available = credits =/= 0.U
  val sent      = io.in.valid && available

  io.in.ready  := available
  io.out.valid := sent
  io.out.bits  := io.in.bits

  when(io.out.credit && !sent) {
    assert(credits < p.linkBufferDepth.U)
    credits := credits + 1.U
  }.elsewhen(sent && !io.out.credit) {
    credits := credits - 1.U
  }
}

@instantiable
class MeshCreditToDecoupled(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new Bundle {
    val in  = Flipped(new MeshCreditPort(p))
    val out = Decoupled(new MeshPacket(p))
  })

  val buffer = Instantiate(new Queue(new MeshPacket(p), p.linkBufferDepth, false, false))
  buffer.io.enq.valid := io.in.valid
  buffer.io.enq.bits  := io.in.bits
  io.out <> buffer.io.deq
  io.in.credit        := buffer.io.deq.fire

  when(io.in.valid) {
    assert(buffer.io.enq.ready, "Mesh credit sender exceeded the receiver capacity")
  }
}
