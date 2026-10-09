package sims.verilator

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{Device, System, SystemParams}
import sims.soc.SystemTarget

class OmniVerilatorTarget extends SystemTarget("../examples/chips/omni/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] = Instantiate(new Device(p.toGlobalConfig))
}
