package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import memcore.memory.bank.{BankParams, BankRequest, BankResponse}

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

class MeshLocalBankRequest(
  addressBits: Int,
  bankBits:    Int,
  dataBits:    Int,
  tagBits:     Int)
    extends BankRequest(BankParams(addressBits, dataBits, tagBits)) {
  val bank = UInt(bankBits.W)
}

/** SharedMem initiates both the source read and target write. */
class MeshLocalBankPort(
  addressBits: Int,
  bankBits:    Int,
  dataBits:    Int,
  tagBits:     Int)
    extends Bundle {
  private val bankParams = BankParams(addressBits, dataBits, tagBits)
  val request            = Decoupled(new MeshLocalBankRequest(addressBits, bankBits, dataBits, tagBits))
  val response           = Flipped(Decoupled(new BankResponse(bankParams)))
}
