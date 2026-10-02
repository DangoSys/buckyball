package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._

/** Core-facing shared-bank port. One request may be outstanding per channel. */
class MeshChannel(p: MeshSharedMemParams) extends Bundle {
  val request  = Flipped(Decoupled(new MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits)))
  val response = Decoupled(new MeshEventBeat(p.global, p.addressBits, p.bankBits, p.tagBits))
}
