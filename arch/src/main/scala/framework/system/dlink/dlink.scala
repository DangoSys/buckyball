package framework.system.dlink

import chisel3._
import framework.system.SystemParams

/** Unified memory access; address decoding belongs to the platform. */
class DLinkIO(p: SystemParams) extends Bundle {
  val axi = new memcore.bus.axi4.Port(p.ddr.axi)
}

trait HasDLink { def dlink: DLinkIO }
