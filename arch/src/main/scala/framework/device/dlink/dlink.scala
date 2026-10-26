package framework.device.dlink

import framework.ip.bus.axi4.{Params => ModuleParam}

import chisel3._
import framework.ip.bus.axi4

/** Unified memory access; address decoding belongs to the device. */
class DLinkIO(b: ModuleParam) extends Bundle {
  val p = b

  val axi = new framework.ip.bus.axi4.Port(b)
}
