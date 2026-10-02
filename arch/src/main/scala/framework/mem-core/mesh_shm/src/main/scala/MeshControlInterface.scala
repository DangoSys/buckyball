package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

/** One full-width word from a Core-local Bank to another Core-local Bank. */
class MeshTransferCommand(p: MeshSharedMemParams) extends Bundle {
  val sourceCore = UInt(p.coreBits.W)
  val sourceBank = UInt(p.localBankBits.W)
  val sourceAddr = UInt(p.addressBits.W)
  val targetCore = UInt(p.coreBits.W)
  val targetBank = UInt(p.localBankBits.W)
  val targetAddr = UInt(p.addressBits.W)
  val tag        = UInt(p.tagBits.W)
}

class MeshTransferCompletion(p: MeshSharedMemParams) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

/** SharedMem initiates both the source read and target write. */
class MeshLocalBankPort(
  b:           GlobalConfig,
  addressBits: Int,
  bankBits:    Int,
  tagBits:     Int)
    extends Bundle {

  val request  = Decoupled(new MeshEventBeat(b, addressBits, bankBits, tagBits))
  val response = Flipped(Decoupled(new MeshEventBeat(b, addressBits, bankBits, tagBits)))
}
