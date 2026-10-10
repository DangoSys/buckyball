package sims.p2e

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

/** Pebble on the P2E board: one compute core with private banks over the DDR4 macro. */
class PebbleP2ETarget extends P2ETarget("../examples/chips/pebble/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
