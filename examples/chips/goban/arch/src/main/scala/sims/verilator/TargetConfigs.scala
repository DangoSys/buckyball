package sims.verilator

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

import sims.soc.SystemTarget

/** Goban on the explicit System: Linux scheduler core 0 and four compute cores. */
class GobanVerilatorTarget extends SystemTarget("../examples/chips/goban/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
