package framework.system.configloader

import framework.system.core.boom.configs.BoomCpuParam
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.tile.configs.TileParam
import framework.top.GlobalConfig

/** Chip topology loaded from chip.pb: the main tile first, then homogeneous compute tiles. */
case class ExampleTopology(tiles: Seq[TileTopology]) {
  require(tiles.nonEmpty && tiles.head.main && tiles.tail.forall(!_.main), "Tile 0 alone is the main tile")
  def mainTile:     TileTopology      = tiles.head
  def computeTiles: Seq[TileTopology] = tiles.tail
}

sealed trait TileCore

case class RocketTileCore(
  cpu:       RocketCpuParam,
  buckyball: Option[GlobalConfig])
    extends TileCore

case class BoomTileCore(cpu: BoomCpuParam) extends TileCore

/**
 * Per-tile topology: ordered cores. CPU kind is per-core; shared memory is tile-level.
 * Main tile cores are Linux visible unless the tile has a controller, in which case
 * only core 0 is visible. Other controller-tile cores are hidden task workers.
 */
case class TileTopology(
  param:      TileParam,
  main:       Boolean,
  cores:      Seq[TileCore],
  hartIds:    Seq[Int],
  signatures: Seq[BigInt],
  controller: Option[Int]) {
  require(main || controller.nonEmpty, "A compute tile needs a controller")
}
