package framework.system.core.rocket

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{CoreBeatBytes, CorePAddrBits, CpuTileParams, HasCoreParameters, MaxHartIdBits, TileKey}
import freechips.rocketchip.rocket.ASIdBits
import freechips.rocketchip.subsystem.CacheBlockBytes

object RocketParameters {

  def apply(cpu: CpuParams): Parameters = Parameters.empty.alterPartial {
    case TileKey         => CpuTileParams(core = cpu.core, dcache = cpu.dcache, icache = cpu.icache, btb = cpu.btb)
    case CorePAddrBits   => cpu.physicalAddressBits
    case MaxHartIdBits   => cpu.hartIdBits
    case CoreBeatBytes   => cpu.beatBytes
    case CacheBlockBytes => cpu.blockBytes
    case ASIdBits        => cpu.asIdBits
  }

}

trait HasCpuParameters extends HasCoreParameters {
  implicit val cpuParams: CpuParams
  implicit lazy val p: Parameters = RocketParameters(cpuParams)
}
