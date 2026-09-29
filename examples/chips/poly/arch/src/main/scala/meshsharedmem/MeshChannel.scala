package examples.poly.meshsharedmem

import chisel3._
import chisel3.util._

/** Core-facing shared-bank port. One request may be outstanding per channel. */
class MeshChannel(p: MeshSharedMemParams) extends Bundle {
  val request  = Flipped(Decoupled(new MeshClientRequest(p)))
  val response = Decoupled(new MeshClientResponse(p))
}
