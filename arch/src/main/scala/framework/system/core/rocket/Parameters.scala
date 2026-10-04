package framework.system.core.rocket

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{CorePAddrBits, HasCoreParameters, MaxHartIdBits, RocketTileParams, TileKey}
import freechips.rocketchip.rocket.ASIdBits
import freechips.rocketchip.subsystem.{CacheBlockBytes, SystemBusKey, SystemBusParams}

object RocketParameters {

  def apply(cpu: CpuParams): Parameters = Parameters.empty.alterPartial {
    case TileKey         => RocketTileParams(core = cpu.core, dcache = cpu.dcache, icache = cpu.icache, btb = cpu.btb)
    case CorePAddrBits   => cpu.physicalAddressBits
    case MaxHartIdBits   => cpu.hartIdBits
    case SystemBusKey    => SystemBusParams(beatBytes = cpu.beatBytes, blockBytes = cpu.blockBytes)
    case CacheBlockBytes => cpu.blockBytes
    case ASIdBits        => cpu.asIdBits
  }

}

trait HasCpuParameters extends HasCoreParameters {
  implicit val cpuParams: CpuParams
  implicit lazy val p: Parameters = RocketParameters(cpuParams)
}
