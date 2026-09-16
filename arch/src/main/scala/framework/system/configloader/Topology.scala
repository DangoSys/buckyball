package framework.system.configloader

import framework.system.tile.PrivateDCacheParams
import framework.system.core.rocket.configs.RocketCoreParam
import framework.system.core.boom.configs.BoomCoreParam
import framework.top.GlobalConfig

/** Loader-private bundle of tile-shared memory fields. */
private[configloader] case class SharedMemFields(
  sharedEnable:            Boolean,
  sharedEntries:           Int,
  sharedInputChannels:     Int,
  sharedDefaultGroupCount: Int)

/** Top-level example topology loaded from chip.pb. */
case class ExampleTopology(tiles: Seq[TileTopology])

sealed trait TileCore

case class RocketTileCore(
  rocket:    RocketCoreParam,
  buckyball: Option[GlobalConfig])
    extends TileCore

case class BoomTileCore(boom: BoomCoreParam) extends TileCore

/**
 * Per-tile topology: ordered cores + optional privateDCache.
 *
 * CPU kind is per-core. Shared mem / private DCache are tile-level.
 */
case class TileTopology(
  cores:         Seq[TileCore],
  privateDCache: Option[PrivateDCacheParams])
