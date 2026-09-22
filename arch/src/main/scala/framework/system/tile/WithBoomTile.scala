package framework.system.tile

import org.chipsalliance.cde.config._
import freechips.rocketchip.subsystem._
import boom.v3.common.{BoomTileAttachParams, BoomTileParams, WithTAGELBPD}
import framework.system.core.boom.configs.BoomCpuParam

/** Attach one BoomTile from BoomCpuParam. */
class WithBoomTile(param: BoomCpuParam, location: HierarchicalLocation = InSubsystem)
    extends Config(
      new WithTAGELBPD ++
        new Config((site, here, up) => {
          case TilesLocated(`location`) =>
            val prev       = up(TilesLocated(`location`), site)
            val idOffset   = up(NumTiles)
            val rowBits    = site(SystemBusKey).beatBits
            val tileParams = BoomTileParams(
              core = BoomCpuParam.toBoomCoreParams(param),
              dcache = Some(BoomCpuParam.toDCacheParams(param, rowBits)),
              icache = Some(BoomCpuParam.toICacheParams(param, rowBits)),
              tileId = idOffset
            )
            BoomTileAttachParams(
              tileParams = tileParams,
              crossingParams = WithNBBTiles.defaultCrossing(location)
            ) +: prev
          case NumTiles                 => up(NumTiles) + 1
        })
    )
