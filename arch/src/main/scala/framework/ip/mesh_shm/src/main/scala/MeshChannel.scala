package framework.ip.memory.mesh_shm

import chisel3._
import chisel3.util._

/** Thread-facing shared-bank port. Tagged requests support bounded outstanding transactions per channel. */
class MeshChannel(p: MeshSharedMemParams) extends Bundle {
  val request  = Flipped(Decoupled(new MeshClientRequest(p)))
  val response = Decoupled(new MeshClientResponse(p))
}
