package framework.device.configs

import buckyball.config.DeviceConfig

import framework.core.configs.CoreParam
import framework.device.ddr.{Params => DdrParams}
import framework.device.peripherals.bootrom.configs.BootRomParams
import framework.device.peripherals.clint.configs.ClintParams
import framework.device.peripherals.plic.configs.PlicParams
import framework.device.peripherals.scu.configs.SCUParams

case class DeviceParam(
  cores:   Seq[CoreParam],
  ddr:     DdrParams,
  bootrom: BootRomParams,
  clint:   ClintParams,
  plic:    PlicParams,
  scu:     SCUParams) {
  val hartIds = cores.map(_.hartId)
  require(hartIds.distinct.size == hartIds.size, "Duplicate core hart ID")
  require(
    clint.hartIds == hartIds && plic.hartIds == hartIds,
    "Peripheral interrupt ordering must match the configured core instances"
  )
}

object DeviceParam {

  def fromProto(p: DeviceConfig, cores: Seq[CoreParam]): DeviceParam = {
    require(
      p.hasDdr && p.hasBootrom && p.hasClint && p.hasPlic && p.hasScu && p.isInitialized,
      "Device configuration is incomplete"
    )
    require(p.getCoresCount == cores.size, "Device core configuration is incomplete")
    DeviceParam(
      cores,
      DdrParams.fromProto(p.getDdr),
      BootRomParams.fromProto(p.getBootrom),
      ClintParams.fromProto(p.getClint),
      PlicParams.fromProto(p.getPlic),
      SCUParams.fromProto(p.getScu)
    )
  }

}
