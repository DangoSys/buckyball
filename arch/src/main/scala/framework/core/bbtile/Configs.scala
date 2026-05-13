package framework.core.bbtile

import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tile.{FPUParams, RocketTileBoundaryBufferParams}
import framework.top.GlobalConfig
import framework.core.bbtile.configs.RocketCoreParam

/**
 * Config fragment to add N BBTiles.
 *
 * Each BBTile can host nCoresPerTile Buckyball slots.
 */
object WithNBBTiles {

  def defaultCrossing(location: HierarchicalLocation): RocketCrossingParams =
    RocketCrossingParams(
      master = HierarchicalElementMasterPortParams.locationDefault(location),
      slave = HierarchicalElementSlavePortParams.locationDefault(location),
      mmioBaseAddressPrefixWhere = location match {
        case InSubsystem          => CBUS
        case InCluster(clusterId) => CCBUS(clusterId)
      }
    )

  /**
   * Resolve the per-tile rocketCore param from buckyballPerCore.
   *
   * Multi-core within a tile currently shares one RocketCoreParam (BBTileParams
   * has tile-level core/dcache/icache fields). All Some(_) entries must agree.
   * Falls back to RocketCoreParam() when nothing is defined.
   */
  def resolveRocketCore(
    buckyballPerCore: Seq[Option[GlobalConfig]],
    buckyballConfig:  GlobalConfig
  ): RocketCoreParam = {
    val defined = buckyballPerCore.flatten.map(_.rocketCore)
    if (defined.isEmpty) {
      buckyballConfig.rocketCore
    } else {
      val head = defined.head
      require(
        defined.forall(_ == head),
        "All cores within a BBTile must currently share the same rocketCore config; " +
          "heterogeneous per-core Rocket params is not yet supported."
      )
      head
    }
  }

}

class WithBBTile(
  location:         HierarchicalLocation = InSubsystem,
  withBuckyball:    Boolean = true,
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  crossing:         Option[RocketCrossingParams] = None,
  nCoresPerTile:    Int = 1,
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None,
  l2cache:          Option[L2CacheParams] = None)
    extends Config((site, here, up) => {
      case TilesLocated(`location`) =>
        val prev                     = up(TilesLocated(`location`), site)
        val idOffset                 = up(NumTiles)
        val actualCrossing           = crossing.getOrElse(WithNBBTiles.defaultCrossing(location))
        val resolvedBuckyballPerCore = buckyballPerCore.getOrElse(
          Seq.fill(nCoresPerTile)(if (withBuckyball) Some(buckyballConfig) else None)
        )
        require(
          resolvedBuckyballPerCore.size == nCoresPerTile,
          s"buckyballPerCore size (${resolvedBuckyballPerCore.size}) must equal nCoresPerTile ($nCoresPerTile)"
        )
        val rocketCore               = WithNBBTiles.resolveRocketCore(resolvedBuckyballPerCore, buckyballConfig)
        val rowBits                  = site(SystemBusKey).beatBits
        val blockBytes               = site(CacheBlockBytes)
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          l2cache = l2cache,
          core = RocketCoreParam.toRocketCoreParams(rocketCore),
          dcache = Some(RocketCoreParam.toDCacheParams(rocketCore, rowBits, blockBytes)),
          icache = Some(RocketCoreParam.toICacheParams(rocketCore, rowBits, blockBytes)),
          btb = RocketCoreParam.toBTBParams(rocketCore)
        )
        BBTileAttachParams(
          tileParams.copy(tileId = idOffset),
          actualCrossing
        ) +: prev
      case NumTiles                 => up(NumTiles) + 1
    })

class WithNBBTiles(
  n:                Int,
  location:         HierarchicalLocation = InSubsystem,
  withBuckyball:    Boolean = true,
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  crossing:         Option[RocketCrossingParams] = None,
  nCoresPerTile:    Int = 1,
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None,
  l2cache:          Option[L2CacheParams] = None)
    extends Config((site, here, up) => {
      case TilesLocated(`location`) =>
        val prev                     = up(TilesLocated(`location`), site)
        val idOffset                 = up(NumTiles)
        val actualCrossing           = crossing.getOrElse(WithNBBTiles.defaultCrossing(location))
        val resolvedBuckyballPerCore = buckyballPerCore.getOrElse(
          Seq.fill(nCoresPerTile)(if (withBuckyball) Some(buckyballConfig) else None)
        )
        require(
          resolvedBuckyballPerCore.size == nCoresPerTile,
          s"buckyballPerCore size (${resolvedBuckyballPerCore.size}) must equal nCoresPerTile ($nCoresPerTile)"
        )
        val rocketCore               = WithNBBTiles.resolveRocketCore(resolvedBuckyballPerCore, buckyballConfig)
        val rowBits                  = site(SystemBusKey).beatBits
        val blockBytes               = site(CacheBlockBytes)
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          l2cache = l2cache,
          core = RocketCoreParam.toRocketCoreParams(rocketCore),
          dcache = Some(RocketCoreParam.toDCacheParams(rocketCore, rowBits, blockBytes)),
          icache = Some(RocketCoreParam.toICacheParams(rocketCore, rowBits, blockBytes)),
          btb = RocketCoreParam.toBTBParams(rocketCore)
        )
        List.tabulate(n)(i =>
          BBTileAttachParams(
            tileParams.copy(tileId = i + idOffset),
            actualCrossing
          )
        ) ++ prev
      case NumTiles                 => up(NumTiles) + n
    })
