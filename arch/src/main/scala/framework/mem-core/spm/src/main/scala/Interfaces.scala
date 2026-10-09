package memcore.memory.spm

import chisel3._
import chisel3.util._
import memcore.memory.bank.BankSetParams

/** One local byte-address window. This does not describe a DDR mapping. */
case class Params(base: BigInt, bytes: Int, dataBits: Int = 128) {
  require(dataBits >= 64 && dataBits % 8 == 0 && isPow2(dataBits / 8))
  val beatBytes = dataBits / 8
  require(base >= 0 && base % beatBytes == 0 && base + bytes <= (BigInt(1) << 64))
  require(bytes >= 2 * beatBytes && isPow2(bytes))
  val laneBits = log2Ceil(beatBytes)
  val sizeBits = log2Ceil(laneBits + 2)
  val bank     = BankSetParams(dataBits, banks = 1, entriesPerBank = bytes / beatBytes)
}

/** Naturally aligned 1..beatBytes access. Data and mask start at address, not the SRAM row. */
class Request(p: Params) extends Bundle {
  val address = UInt(64.W)
  val size    = UInt(p.sizeBits.W)
  val write   = Bool()
  val data    = UInt(p.dataBits.W)
  val mask    = UInt(p.beatBytes.W)
}

class Response(p: Params) extends Bundle {
  val data  = UInt(p.dataBits.W)
  val error = Bool()
}

class Port(p: Params) extends Bundle {
  val request  = Decoupled(new Request(p))
  val response = Flipped(Decoupled(new Response(p)))
}
