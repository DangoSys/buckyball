package examples.designs.poly.sims.device

import buckyball.config.Design
import chisel3.experimental.hierarchy.{Instance, Instantiate}
import examples.designs.poly.DeviceTop
import framework.device.cbus.CBus
import framework.device.cbus.generated.CoreComponents
import framework.device.dlink.BaseDevice
import _root_.configs.GlobalConfig
import _root_.sims.soc.SystemTarget
import java.nio.file.{Files, Paths}

class DeviceConfig(design: Design)
    extends SystemTarget(
      GlobalConfig.fromProto(design, CBus.parameters(design, CoreComponents.factories))
    ) {
  def instantiate(b: GlobalConfig): Instance[BaseDevice] = Instantiate(new DeviceTop(b, design))
}
