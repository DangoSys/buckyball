package sims.verilator

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

import sims.soc.SystemTarget

class ToyVerilatorTarget extends SystemTarget("../examples/chips/toy/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
}
