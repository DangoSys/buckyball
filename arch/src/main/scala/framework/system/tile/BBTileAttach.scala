package framework.system.tile

import freechips.rocketchip.subsystem.{CanAttachTile, RocketCrossingParams}
import freechips.rocketchip.tile.RocketTile

/** Attach parameters for BBTile — used in chipyard Config system via TilesLocated. */
case class BBTileAttachParams(
  tileParams:     BBTileParams,
  crossingParams: RocketCrossingParams)
    extends CanAttachTile { type TileType = BBTile }
