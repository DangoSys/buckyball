package sims.p2e

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

/** Goban on the P2E board: Linux scheduler core 0 and four compute cores. */
class GobanP2ETarget extends P2ETarget("../examples/chips/goban/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
