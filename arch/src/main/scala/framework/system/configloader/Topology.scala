package framework.system.configloader

import framework.system.tile.PrivateDCacheParams
import framework.system.core.configs.RocketCoreParam
import framework.top.GlobalConfig

/** Loader-private bundle of tile-shared memory fields. */
private[configloader] case class SharedMemFields(
  sharedEnable:            Boolean,
  sharedEntries:           Int,
  sharedInputChannels:     Int,
  sharedDefaultGroupCount: Int)

/** Top-level example topology loaded from chip.pb. */
case class ExampleTopology(tiles: Seq[TileTopology])

/**
 * Per-tile topology: cores + optional privateDCache.
 *
 * @param cores         One entry per core in this tile. `None` disables the
 *                      Buckyball slot for that core (Rocket-only).
 * @param privateDCache Resolved per-tile private DCache parameters, or `None`
 *                      to skip the private DCache layer entirely.
 */
case class TileTopology(
  cores:         Seq[Option[GlobalConfig]],
  privateDCache: Option[PrivateDCacheParams],
  rocketCores:   Seq[RocketCoreParam])
