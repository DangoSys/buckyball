package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._

/** Core-facing shared-bank port. Non-conflicting requests may be outstanding together. */
class MeshChannel(p: MeshSharedMemParams) extends Bundle {
  val request  = Flipped(Decoupled(new MeshClientRequest(p)))
  val response = Decoupled(new MeshClientResponse(p))
}
