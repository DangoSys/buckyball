package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._

/** Contiguous rows from one Core-local Bank to another Core-local Bank. */
class MeshTransferCommand(p: MeshSharedMemParams) extends Bundle {
  val sourceCore = UInt(8.W)
  val sourceBank = UInt(p.localBankBits.W)
  val sourceAddr = UInt(16.W)
  val targetCore = UInt(8.W)
  val targetBank = UInt(p.localBankBits.W)
  val targetAddr = UInt(16.W)
  val rows       = UInt(17.W)
  val tag        = UInt(p.tagBits.W)
}

class MeshTransferCompletion(p: MeshSharedMemParams) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

class MeshLocalBankRequest(
  addressBits: Int,
  bankBits:    Int,
  dataBits:    Int,
  tagBits:     Int)
    extends Bundle {
  val addr  = UInt(addressBits.W)
  val write = Bool()
  val data  = UInt(dataBits.W)
  val mask  = UInt((dataBits / 8).W)
  val tag   = UInt(tagBits.W)
  val bank  = UInt(bankBits.W)
}

class MeshLocalBankResponse(dataBits: Int, tagBits: Int) extends Bundle {
  val data  = UInt(dataBits.W)
  val tag   = UInt(tagBits.W)
  val error = Bool()
}

/** SharedMem initiates both the source read and target write. */
class MeshLocalBankPort(
  addressBits: Int,
  bankBits:    Int,
  dataBits:    Int,
  tagBits:     Int)
    extends Bundle {

  val request  = Decoupled(new MeshLocalBankRequest(addressBits, bankBits, dataBits, tagBits))
  val response = Flipped(Decoupled(new MeshLocalBankResponse(dataBits, tagBits)))
}
