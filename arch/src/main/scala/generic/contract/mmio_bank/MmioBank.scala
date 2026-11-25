package framework.generic.contract.mmio_bank

import framework.generic.arbiter.{Arbiter, RRArbiter}
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.mmio_bank.MetadataWrite

/** Metadata receiver: one addressed DMA beat, acknowledged after every byte is stored. */
class MetadataWrite(bankWidth: Int, mmioBytes: Int) extends Bundle {

  val request = Flipped(Decoupled(new Bundle {
    val addr = UInt(log2Ceil(mmioBytes).W)
    val data = UInt(bankWidth.W)
    val mask = UInt((bankWidth / 8).W)
  }))

  val response = Decoupled(Bool())
}
