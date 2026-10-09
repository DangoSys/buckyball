package framework.system

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.configloader.ExampleTopology
import framework.system.device.DeviceParams
import framework.system.dlink.{DLinkIO, HasDLink}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.ddr.{Params => DdrParams}

case class SystemParams(
  topology:        ExampleTopology,
  cpuPhysicalBits: Int,
  memory:          CoherenceParams,
  l1:              RnfParams,
  regions:         Seq[PhysicalRegion],
  tracking:        TrackingParams,
  ddr:             DdrParams,
  devices:         DeviceParams = DeviceParams()) {

  def toGlobalConfig: framework.top.GlobalConfig = GlobalConfigOps.fromSystem(this)

}

@instantiable
abstract class System(p: SystemParams) extends Module with HasDLink {
  @public val dlink = IO(new DLinkIO(p))
}
