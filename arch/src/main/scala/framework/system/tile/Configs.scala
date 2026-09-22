package framework.system.tile

import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tile.{FPUParams, RocketTileBoundaryBufferParams}
import framework.top.GlobalConfig
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.tile.configs.TileParam

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

  def resolveRocketCpus(
    nCoresPerTile:    Int,
    rocketCpuPerCore: Option[Seq[RocketCpuParam]]
  ): Seq[RocketCpuParam] = {
    rocketCpuPerCore match {
      case Some(cores) =>
        require(
          cores.size == nCoresPerTile,
          s"rocketCpuPerCore size (${cores.size}) must equal nCoresPerTile ($nCoresPerTile)"
        )
        cores
      case None        =>
        throw new RuntimeException("rocketCpuPerCore must be specified")
    }
  }

}

class WithBBTile(
  tileParam:        TileParam,
  location:         HierarchicalLocation = InSubsystem,
  withBuckyball:    Boolean = true,
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  crossing:         Option[RocketCrossingParams] = None,
  nCoresPerTile:    Int = 1,
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None,
  rocketCpuPerCore: Option[Seq[RocketCpuParam]] = None,
  privateDCache:    Option[PrivateDCacheParams] = None,
  hiddenHartBase:   Option[Int] = None)
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
        val rocketCpus               = WithNBBTiles.resolveRocketCpus(
          nCoresPerTile,
          rocketCpuPerCore
        )
        val rocketCpu                = rocketCpus.head
        val rowBits                  = site(SystemBusKey).beatBits
        val blockBytes               = site(CacheBlockBytes)
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          rocketCorePerCore = rocketCpus.map(RocketCpuParam.toRocketCoreParams(_, tileParam.xLen, tileParam.pgLevels)),
          privateDCache = privateDCache,
          hiddenHartBase = hiddenHartBase,
          core = RocketCpuParam.toRocketCoreParams(rocketCpu, tileParam.xLen, tileParam.pgLevels),
          dcache = Some(RocketCpuParam.toDCacheParams(rocketCpu, rowBits, blockBytes)),
          icache = Some(RocketCpuParam.toICacheParams(rocketCpu, rowBits, blockBytes)),
          btb = RocketCpuParam.toBTBParams(rocketCpu)
        )
        BBTileAttachParams(
          tileParams.copy(tileId = idOffset),
          actualCrossing
        ) +: prev
      case NumTiles                 => up(NumTiles) + 1
    })

class WithNBBTiles(
  n:                Int,
  tileParam:        TileParam,
  location:         HierarchicalLocation = InSubsystem,
  withBuckyball:    Boolean = true,
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  crossing:         Option[RocketCrossingParams] = None,
  nCoresPerTile:    Int = 1,
  buckyballPerCore: Option[Seq[Option[GlobalConfig]]] = None,
  rocketCpuPerCore: Option[Seq[RocketCpuParam]] = None,
  privateDCache:    Option[PrivateDCacheParams] = None,
  hiddenHartBase:   Option[Int] = None)
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
        val rocketCpus               = WithNBBTiles.resolveRocketCpus(
          nCoresPerTile,
          rocketCpuPerCore
        )
        val rocketCpu                = rocketCpus.head
        val rowBits                  = site(SystemBusKey).beatBits
        val blockBytes               = site(CacheBlockBytes)
        val tileParams               = BBTileParams(
          nCores = nCoresPerTile,
          withBuckyball = withBuckyball,
          buckyballConfig = buckyballConfig,
          buckyballPerCore = resolvedBuckyballPerCore,
          rocketCorePerCore = rocketCpus.map(RocketCpuParam.toRocketCoreParams(_, tileParam.xLen, tileParam.pgLevels)),
          privateDCache = privateDCache,
          hiddenHartBase = hiddenHartBase,
          core = RocketCpuParam.toRocketCoreParams(rocketCpu, tileParam.xLen, tileParam.pgLevels),
          dcache = Some(RocketCpuParam.toDCacheParams(rocketCpu, rowBits, blockBytes)),
          icache = Some(RocketCpuParam.toICacheParams(rocketCpu, rowBits, blockBytes)),
          btb = RocketCpuParam.toBTBParams(rocketCpu)
        )
        List.tabulate(n)(i =>
          BBTileAttachParams(
            tileParams.copy(tileId = i + idOffset),
            actualCrossing
          )
        ) ++ prev
      case NumTiles                 => up(NumTiles) + n
    })
