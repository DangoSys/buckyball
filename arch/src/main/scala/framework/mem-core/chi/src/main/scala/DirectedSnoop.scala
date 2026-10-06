package memcore.bus.chi

import chisel3._

class DirectedSnoop(p: Params) extends Bundle {
  val targetNode = UInt(p.nodeIdBits.W)
  val flit       = new SnoopFlit(p)
}
