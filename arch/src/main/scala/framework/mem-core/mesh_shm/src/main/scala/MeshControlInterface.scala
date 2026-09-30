package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import memcore.memory.bank.{BankParams, BankRequest, BankResponse}

/** One full-width word from a Core-local Bank to another Core-local Bank. */
class MeshTransferCommand(p: MeshSharedMemParams) extends Bundle {
  val sourceCore = UInt(p.coreBits.W)
  val sourceAddr = UInt(p.addressBits.W)
  val targetCore = UInt(p.coreBits.W)
  val targetAddr = UInt(p.addressBits.W)
  val tag        = UInt(p.tagBits.W)
}

class MeshTransferCompletion(p: MeshSharedMemParams) extends Bundle {
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

/** SharedMem initiates both the source read and target write. */
class MeshLocalBankPort(p: MeshSharedMemParams) extends Bundle {
  private val bankParams = BankParams(p.addressBits, p.dataBits, p.tagBits)
  val request            = Decoupled(new BankRequest(bankParams))
  val response           = Flipped(Decoupled(new BankResponse(bankParams)))
}
