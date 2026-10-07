package framework.system.core.rocket

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{
  CacheBlockBytes,
  CorePAddrBits,
  HasCoreParameters,
  MasterPortBeatBytes,
  MaxHartIdBits,
  TileKey,
  TileParams
}
import freechips.rocketchip.rocket.ASIdBits

object RocketParameters {

  def apply(cpu: CpuParams): Parameters = Parameters.empty.alterPartial {
    case TileKey             => TileParams(core = cpu.core, dcache = cpu.dcache, icache = cpu.icache, btb = cpu.btb)
    case CorePAddrBits       => cpu.physicalAddressBits
    case MaxHartIdBits       => cpu.hartIdBits
    case MasterPortBeatBytes => cpu.beatBytes
    case CacheBlockBytes     => cpu.blockBytes
    case ASIdBits            => cpu.asIdBits
  }

}

trait HasCpuParameters extends HasCoreParameters {
  implicit val cpuParams: CpuParams
  implicit lazy val p: Parameters = RocketParameters(cpuParams)
}
