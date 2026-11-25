package examples.designs.gemmini.devices.gemmini.sims.device

import buckyball.config.Design
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import examples.designs.gemmini.devices.gemmini.DeviceTop
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import framework.device.dlink.BaseDevice
import framework.device.sims.configs.GlobalConfig
import framework.device.sims.soc.SystemTarget

class DeviceConfig(design: Design)
    extends SystemTarget(
      GlobalConfig.fromProto(design, CBus.parameters(design, HartComponents.factories))
    ) {
  def instantiate(b: GlobalConfig): Instance[BaseDevice] = Instantiate(new DeviceTop(b, design))
  def hierBlocks = Seq()
}
