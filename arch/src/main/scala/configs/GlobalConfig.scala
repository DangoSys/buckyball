package configs

import buckyball.config.Design
import framework.core.configs.CoreParam
import scala.jdk.CollectionConverters._

import chisel3.experimental.SerializableModuleParameter
import framework.device.configs.DeviceParam
import sims.configs.SimParam

case class GlobalConfig(
  device: DeviceParam,
  sim:    SimParam)
    extends SerializableModuleParameter

object GlobalConfig {

  def fromProto(design: Design, cores: Seq[CoreParam]): GlobalConfig = {
    require(design.hasDevice && design.hasSims, "Design device and simulation configuration are required")
    GlobalConfig(DeviceParam.fromProto(design.getDevice, cores), SimParam.fromProto(design.getSims))
  }

  def fromProto(design: Design): GlobalConfig = {
    require(design.getThreadsCount == 0, "Thread implementations must supply their core parameters")
    fromProto(design, design.getDevice.getCoresList.asScala.map(p => CoreParam.fromProto(p, Seq.empty)).toSeq)
  }

}
