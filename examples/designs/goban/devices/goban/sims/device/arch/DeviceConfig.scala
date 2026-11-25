package examples.designs.goban.devices.goban.sims.device

import buckyball.config.Design
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import examples.designs.goban.devices.goban.DeviceTop
import framework.hart.cbus.CBus
import framework.hart.cbus.generated.HartComponents
import framework.device.dlink.BaseDevice
import framework.device.sims.configs.GlobalConfig
import framework.device.sims.soc.SystemTarget
import java.nio.file.{Files, Paths}

class DeviceConfig(design: Design)
    extends SystemTarget(
      GlobalConfig.fromProto(design, CBus.parameters(design, HartComponents.factories))
    ) {
  def instantiate(b: GlobalConfig): Instance[BaseDevice] = Instantiate(new DeviceTop(b, design))
  def hierBlocks = Seq()
}
