// See LICENSE.Berkeley and LICENSE.SiFive for Rocket interface definitions.
package freechips.rocketchip.tile {

  import org.chipsalliance.cde.config.{Field, Parameters}
  import chisel3.util.{log2Ceil, log2Up}
  import freechips.rocketchip.rocket._

  case object TileKey       extends Field[CpuTileParams]
  case object CoreBeatBytes extends Field[Int]
  case object CoreRoCCCount extends Field[Int](0)

  trait HasTileParameters {
    implicit val p: Parameters
    def tileParams: CpuTileParams = p(TileKey)
    def usingVM                    = tileParams.core.useVM
    def usingSupervisor            = tileParams.core.hasSupervisorMode
    def usingUser                  = tileParams.core.useUser || usingSupervisor
    def usingHypervisor            = usingVM && tileParams.core.useHypervisor
    def usingDebug                 = tileParams.core.useDebug
    def usingRoCC                  = p(CoreRoCCCount) > 0
    def usingBTB                   = tileParams.btb.exists(_.nEntries > 0)
    def usingPTW                   = usingVM
    def usingDataScratchpad        = tileParams.dcache.flatMap(_.scratch).isDefined
    def xLen                       = tileParams.core.xLen
    def xBytes                     = xLen / 8
    def iLen                       = 32
    def pgIdxBits                  = 12
    def pgLevelBits                = 10 - log2Ceil(xLen / 32)
    def pgLevels                   = tileParams.core.pgLevels
    def minPgLevels = { val levels = if (xLen == 32) 2 else 3; require(pgLevels >= levels); levels }
    def maxSVAddrBits              = pgIdxBits + pgLevels * pgLevelBits
    def maxHypervisorExtraAddrBits = 2
    def hypervisorExtraAddrBits    = if (usingHypervisor) maxHypervisorExtraAddrBits else 0
    def maxHVAddrBits              = maxSVAddrBits + hypervisorExtraAddrBits
    def asIdBits                   = p(ASIdBits)
    def vmIdBits                   = p(VMIdBits)

    lazy val maxPAddrBits: Int = {
      require(xLen == 32 || xLen == 64); if (!usingVM) xLen else if (xLen == 32) 34 else 56
    }

    lazy val paddrBits: Int = p(CorePAddrBits)

    def vaddrBits: Int =
      if (usingVM) {
        val bits = maxHVAddrBits
        require(bits == xLen || xLen > bits && bits > paddrBits)
        bits
      } else (paddrBits + 1).min(xLen)

    def vpnBits             = vaddrBits - pgIdxBits
    def ppnBits             = paddrBits - pgIdxBits
    def vpnBitsExtended     = vpnBits + (if (vaddrBits < xLen) 1 + (if (usingHypervisor) 1 else 0) else 0)
    def vaddrBitsExtended   = vpnBitsExtended + pgIdxBits
    def cacheBlockBytes     = p(freechips.rocketchip.subsystem.CacheBlockBytes)
    def lgCacheBlockBytes   = log2Up(cacheBlockBytes)
    def masterPortBeatBytes = p(CoreBeatBytes)
    def dcacheArbPorts      = 1 + (if (usingVM) 1 else 0) + (if (usingDataScratchpad) 1 else 0) + p(CoreRoCCCount) +
      (if (tileParams.core.useVector && tileParams.core.vectorUseDCache) 1 else 0)
  }

}

package freechips.rocketchip.rocket {
  import org.chipsalliance.cde.config.Field
  case object ASIdBits extends Field[Int](0)
  case object VMIdBits extends Field[Int](0)
}

package freechips.rocketchip.subsystem {
  import org.chipsalliance.cde.config.Field
  case object CacheBlockBytes extends Field[Int]
}

package freechips.rocketchip.devices.debug {
  import org.chipsalliance.cde.config.Field
  case object DebugModuleKey extends Field[Option[DebugModuleParams]](Some(DebugModuleParams()))
}
