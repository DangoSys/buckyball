package memcore.bus.chi

import chisel3._
import chisel3.util._

case class HomeMapping(count: Int, base: Int = 64) {
  require(count >= 1 && isPow2(count))
  val stripeBits = log2Ceil(count)
  def node(address:         UInt): UInt = base.U + ((address >> 6) & (count - 1).U)
  def localAddress(address: UInt): UInt =
    if (count == 1) address else Cat(address(address.getWidth - 1, 6 + stripeBits), address(5, 0))
}
