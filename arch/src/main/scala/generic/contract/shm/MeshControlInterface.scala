package framework.generic.contract.shm

import framework.generic.contract.shm.LocalBank
import framework.generic.contract.shm.MeshLocalBankRequest
import framework.generic.contract.shm.MeshLocalBankResponse

import chisel3._
import chisel3.util._

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
class LocalBank(
  addressBits: Int,
  bankBits:    Int,
  dataBits:    Int,
  tagBits:     Int)
    extends Bundle {

  val request  = Decoupled(new MeshLocalBankRequest(addressBits, bankBits, dataBits, tagBits))
  val response = Flipped(Decoupled(new MeshLocalBankResponse(dataBits, tagBits)))
}
