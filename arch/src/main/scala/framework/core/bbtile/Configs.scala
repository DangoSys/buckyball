package framework.core.bbtile

import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tile.{FPUParams, RocketTileBoundaryBufferParams}
import framework.top.GlobalConfig

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

}

class WithBBTile(
  location:         HierarchicalLocation = InSubsystem,
  withBuckyball:    Boolean = true,
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  crossing:         Option[RocketCrossingParams] = None,
  nCoresPerTile:    Int = 1,
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None)
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
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          core = RocketCoreParams(
            mulDiv = Some(MulDivParams(
              mulUnroll = 8,
              mulEarlyOut = true,
              divEarlyOut = true
            )),
            useZba = true,
            useZbb = true,
            useZbs = true,
            fpu = Some(FPUParams(minFLen = 16))
          ),
          dcache = Some(DCacheParams(
            nSets = 64,
            nWays = 8,
            rowBits = site(SystemBusKey).beatBits,
            nMSHRs = 0,
            blockBytes = site(CacheBlockBytes)
          )),
          icache = Some(ICacheParams(
            nSets = 64,
            nWays = 8,
            rowBits = site(SystemBusKey).beatBits,
            blockBytes = site(CacheBlockBytes)
          ))
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
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None)
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
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          core = RocketCoreParams(
            mulDiv = Some(MulDivParams(
              mulUnroll = 8,
              mulEarlyOut = true,
              divEarlyOut = true
            )),
            useZba = true,
            useZbb = true,
            useZbs = true,
            fpu = Some(FPUParams(minFLen = 16))
          ),
          dcache = Some(DCacheParams(
            nSets = 64,
            nWays = 8,
            rowBits = site(SystemBusKey).beatBits,
            nMSHRs = 0,
            blockBytes = site(CacheBlockBytes)
          )),
          icache = Some(ICacheParams(
            nSets = 64,
            nWays = 8,
            rowBits = site(SystemBusKey).beatBits,
            blockBytes = site(CacheBlockBytes)
          ))
        )
        List.tabulate(n)(i =>
          BBTileAttachParams(
            tileParams.copy(tileId = i + idOffset),
            actualCrossing
          )
        ) ++ prev
      case NumTiles                 => up(NumTiles) + n
    })
