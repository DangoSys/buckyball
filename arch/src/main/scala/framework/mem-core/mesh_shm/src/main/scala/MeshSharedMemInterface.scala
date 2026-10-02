package memcore.memory.mesh_shm

import chisel3._
import memcore.bus.axi.Beat
import framework.top.GlobalConfig

object MeshEvent {
  val ReadRequest   = 0.U(3.W)
  val WriteRequest  = 1.U(3.W)
  val ReadResponse  = 2.U(3.W)
  val WriteResponse = 3.U(3.W)
}

/** One row per AXI-S beat. TUSER[1:0] is the event and TUSER[2] is the error bit. */
class MeshEventBeat(
  b:           GlobalConfig,
  addressBits: Int,
  bankBits:    Int,
  tagBits:     Int)
    extends Beat(b.memDomain.bankWidth, tagBits, bankBits, 3) {
  val addr = UInt(addressBits.W)
}

/** Mesh-internal AXI-S+ event with XY routing metadata. */
class MeshPacket(p: MeshSharedMemParams) extends MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits) {
  val destRow   = UInt(p.rowBits.W)
  val destCol   = UInt(p.colBits.W)
  val sourceRow = UInt(p.rowBits.W)
  val sourceCol = UInt(p.colBits.W)
  val channel   = UInt(p.channelBits.W)
}
