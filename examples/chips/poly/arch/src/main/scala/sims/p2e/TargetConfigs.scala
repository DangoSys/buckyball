package sims.p2e

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

/** Poly on the P2E board: a CPU main tile and eight homogeneous compute tiles. */
class PolyP2ETarget extends P2ETarget("../examples/chips/poly/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
