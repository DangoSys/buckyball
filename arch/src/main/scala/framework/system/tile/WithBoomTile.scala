package framework.system.tile

import org.chipsalliance.cde.config._
import freechips.rocketchip.subsystem._
import boom.v3.common.{BoomTileAttachParams, BoomTileParams, WithTAGELBPD}
import framework.system.core.boom.configs.BoomCoreParam

/** Attach one BoomTile from BoomCoreParam. */
class WithBoomTile(param: BoomCoreParam, location: HierarchicalLocation = InSubsystem)
    extends Config(
      new WithTAGELBPD ++
        new Config((site, here, up) => {
          case TilesLocated(`location`) =>
            val prev       = up(TilesLocated(`location`), site)
            val idOffset   = up(NumTiles)
            val rowBits    = site(SystemBusKey).beatBits
            val tileParams = BoomTileParams(
              core = BoomCoreParam.toBoomCoreParams(param),
              dcache = Some(BoomCoreParam.toDCacheParams(param, rowBits)),
              icache = Some(BoomCoreParam.toICacheParams(param, rowBits)),
              tileId = idOffset
            )
            BoomTileAttachParams(
              tileParams = tileParams,
              crossingParams = WithNBBTiles.defaultCrossing(location)
            ) +: prev
          case NumTiles                 => up(NumTiles) + 1
        })
    )
