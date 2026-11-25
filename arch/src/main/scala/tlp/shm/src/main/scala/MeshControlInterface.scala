package framework.tlp.shm

import chisel3._
import chisel3.util._

/** Contiguous rows from one Thread-local Bank to another Thread-local Bank. */
class MeshTransferCommand(p: MeshSharedMemParams) extends Bundle {
  val sourceThread = UInt(8.W)
  val sourceBank   = UInt(p.localBankBits.W)
  val sourceAddr   = UInt(16.W)
  val targetThread = UInt(8.W)
  val targetBank   = UInt(p.localBankBits.W)
  val targetAddr   = UInt(16.W)
  val rows         = UInt(17.W)
  val tag          = UInt(p.tagBits.W)
}

class MeshTransferCompletion(p: MeshSharedMemParams) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}
