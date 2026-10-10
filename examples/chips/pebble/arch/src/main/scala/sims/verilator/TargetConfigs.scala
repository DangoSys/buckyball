package sims.verilator

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

import sims.soc.SystemTarget

/** Pebble on the explicit System: one compute core with private banks and no task controller. */
class PebbleVerilatorTarget extends SystemTarget("../examples/chips/pebble/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
