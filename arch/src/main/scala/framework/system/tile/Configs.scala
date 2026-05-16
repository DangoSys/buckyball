package framework.system.tile

import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket.RocketCoreParams
import freechips.rocketchip.subsystem._

import framework.top.GlobalConfig
import framework.system.configs.RocketCoreParam
import framework.system.core.BBCoreParams
import framework.system.core.scu.SCUParams
import framework.system.sm.SMParams

/**
 * Helpers for new BBTile config fragments.
 */
object WithBBTile {

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

/**
 * Config fragment to add a single BBTile (Phase 5: single Core, single SM only).
 *
 * Uses the new Core-SM architecture (`framework.system.tile.BBTileParams`).
 *
 * For a no-Buckyball variant, pass `withBuckyball = false`.
 *
 * @param location        Where to attach the tile (default: InSubsystem)
 * @param withBuckyball   Whether to enable Buckyball accelerator (default: true)
 * @param globalConfig    Global config (always required, provides xLen/paddrBits/etc)
 * @param crossing        Optional crossing override; defaults to standard rocket crossing
 */
class WithBBTile(
  location:      HierarchicalLocation = InSubsystem,
  withBuckyball: Boolean = true,
  globalConfig:  GlobalConfig = GlobalConfig(),
  crossing:      Option[RocketCrossingParams] = None)
    extends Config((site, here, up) => {
      case TilesLocated(`location`) =>
        val prev           = up(TilesLocated(`location`), site)
        val idOffset       = up(NumTiles)
        val actualCrossing = crossing.getOrElse(WithBBTile.defaultCrossing(location))

        // Resolve Rocket core and cache parameters from the GlobalConfig
        val rocketCoreParam = globalConfig.rocketCore
        val rowBits         = site(SystemBusKey).beatBits
        val blockBytes      = site(CacheBlockBytes)

        val rocketParams = RocketCoreParam.toRocketCoreParams(rocketCoreParam)
        val icacheParams = RocketCoreParam.toICacheParams(rocketCoreParam, rowBits, blockBytes)
        val dcacheParams = RocketCoreParam.toDCacheParams(rocketCoreParam, rowBits, blockBytes)
        val btbParams    = RocketCoreParam.toBTBParams(rocketCoreParam)

        // Build single SMParams (one SM per Core for now)
        val smParams = SMParams(
          rocket = rocketParams,
          globalConfig = globalConfig,
          hasBuckyball = withBuckyball
        )

        // Build single BBCoreParams (one Core per Tile for now)
        val coreParams = BBCoreParams(
          nSMs = 1,
          smParams = smParams,
          l1ICache = icacheParams,
          l1DCache = dcacheParams,
          btb = btbParams,
          scu = SCUParams(),
          beuAddr = None
        )

        // Build BBTileParams: 1 Core, IOManager with 1 input, no L2 by default.
        val tileParams = BBTileParams(
          cores = Seq(coreParams),
          ioManager = IOManagerParams(nCores = 1),
          l2cache = None,
          tileId = idOffset
        )

        BBTileAttachParams(tileParams, actualCrossing) +: prev

      case NumTiles => up(NumTiles) + 1
    })
