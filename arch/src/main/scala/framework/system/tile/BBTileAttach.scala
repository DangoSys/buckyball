package framework.system.tile

import freechips.rocketchip.subsystem.{CanAttachTile, RocketCrossingParams}

/**
 * Attach parameters for BBTile - used in chipyard Config system via TilesLocated.
 *
 * This case class wraps BBTileParams and crossing parameters, allowing BBTile
 * to be instantiated via the standard Chipyard tile attachment mechanism.
 */
case class BBTileAttachParams(
  tileParams:     BBTileParams,
  crossingParams: RocketCrossingParams)
    extends CanAttachTile {
  type TileType = BBTile
}
