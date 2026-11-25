package framework.device.configs

import buckyball.config.DeviceConfig

import framework.generic.ddr.{Params => DdrParams}
import framework.device.peripherals.bootrom.configs.BootRomParams
import framework.device.peripherals.clint.configs.ClintParams
import framework.device.peripherals.plic.configs.PlicParams
import framework.device.peripherals.scu.configs.SCUParams
import framework.device.peripherals.trace.configs.TraceParams

case class DeviceParam(
  hartIds: Seq[Int],
  ddr:     DdrParams,
  bootrom: BootRomParams,
  clint:   ClintParams,
  plic:    PlicParams,
  scu:     SCUParams,
  trace:   TraceParams) {
  require(hartIds.distinct.size == hartIds.size, "Duplicate hart hart ID")
  require(
    clint.hartIds == hartIds && plic.hartIds == hartIds,
    "Peripheral interrupt ordering must match the configured hart instances"
  )
}

object DeviceParam {

  def fromProto(p: DeviceConfig, hartIds: Seq[Int]): DeviceParam = {
    require(
      p.hasDdr && p.hasBootrom && p.hasClint && p.hasPlic && p.hasScu && p.hasTrace && p.isInitialized,
      "Device configuration is incomplete"
    )
    require(p.getHartsCount == hartIds.size, "Device hart configuration is incomplete")
    DeviceParam(
      hartIds,
      DdrParams.fromProto(p.getDdr),
      BootRomParams.fromProto(p.getBootrom),
      ClintParams.fromProto(p.getClint),
      PlicParams.fromProto(p.getPlic),
      SCUParams.fromProto(p.getScu),
      TraceParams.fromProto(p.getTrace)
    )
  }

}
