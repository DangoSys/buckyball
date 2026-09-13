package memcore.memory.coherence

import chisel3._
import chisel3.util._
import memcore.bus.chi._

case class HomeMapping(count: Int, base: Int = 64) {
  require(count >= 1 && isPow2(count))
  val stripeBits = log2Ceil(count)
  def node(address:         UInt): UInt = base.U + ((address >> 6) & (count - 1).U)
  def localAddress(address: UInt): UInt =
    if (count == 1) address else Cat(address(address.getWidth - 1, 6 + stripeBits), address(5, 0))
}

// The internal protocol boundary; physical CHI channels are attached by the fabric.
class ChiRequesterPort(p: ChiParams) extends Bundle {
  val req   = Decoupled(new ChiReq(p))
  val txRsp = Decoupled(new ChiRsp(p))
  val txDat = Decoupled(new ChiDat(p))
  val snp   = Flipped(Decoupled(new ChiSnp(p)))
  val rxRsp = Flipped(Decoupled(new ChiRsp(p)))
  val rxDat = Flipped(Decoupled(new ChiDat(p)))
}

class CacheAccess(p: ChiParams) extends Bundle {
  val addr   = UInt(p.addressBits.W)
  val write  = Bool()
  val data   = UInt(64.W)
  val mask   = UInt(8.W)
  val atomic = UInt(4.W)
}

object CacheAtomic {
  val None  = 0
  val Swap  = 1
  val Add   = 2
  val Xor   = 3
  val And   = 4
  val Or    = 5
  val Min   = 6
  val Max   = 7
  val MinU  = 8
  val MaxU  = 9
  val LR    = 10
  val SC    = 11
  val Fence = 12
}

class CacheResult extends Bundle {
  val data  = UInt(64.W)
  val error = Bool()
}

class CacheLineState(p: ChiParams) extends Bundle {
  val valid    = Bool()
  val writable = Bool()
  val line     = UInt((p.addressBits - 6).W)
}

object CoherenceState {
  val I         = 0
  val SC        = 1
  val UC        = 2
  val PassDirty = 4
}
