package framework.device.sims.configs

import buckyball.config.Design
import framework.hart.configs.HartParam
import scala.jdk.CollectionConverters._

import chisel3.experimental.SerializableModuleParameter
import framework.device.configs.DeviceParam
import framework.generic.contract.sims.configs.SimParam

case class GlobalConfig(
  device: DeviceParam,
  harts:  Seq[HartParam],
  sim:    SimParam)
    extends SerializableModuleParameter

object GlobalConfig {

  def fromProto(design: Design, harts: Seq[HartParam]): GlobalConfig = {
    require(design.hasDevice && design.hasSims, "Design device and simulation configuration are required")
    GlobalConfig(DeviceParam.fromProto(design.getDevice, harts.map(_.hartId)), harts, SimParam.fromProto(design.getSims))
  }

  def fromProto(design: Design): GlobalConfig = {
    require(design.getThreadsCount == 0, "Thread implementations must supply their hart parameters")
    fromProto(design, design.getDevice.getHartsList.asScala.map(p => HartParam.fromProto(p, Seq.empty)).toSeq)
  }

}
