package framework.system.configloader

import framework.system.core.boom.configs.BoomCpuParam
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.tile.configs.TileParam
import framework.top.GlobalConfig

/** Chip topology loaded from chip.pb: the main tile first, then configured TLink tiles. */
case class ExampleTopology(tiles: Seq[TileTopology]) {
  require(tiles.nonEmpty && tiles.head.main && tiles.tail.forall(!_.main), "Tile 0 alone is the main tile")
  def mainTile:     TileTopology      = tiles.head
  def mountedTiles: Seq[TileTopology] = tiles.tail
}

sealed trait TileCore

case class RocketTileCore(
  cpu:       RocketCpuParam,
  buckyball: Option[GlobalConfig])
    extends TileCore

case class BoomTileCore(cpu: BoomCpuParam) extends TileCore

case class AntTileCore(local: framework.ant.Params, buckyball: GlobalConfig) extends TileCore

case class SharedStorageParams(bankBits: Int, bankEntries: Int, banks: Int) {
  require(bankBits == 128 && bankEntries > 0 && banks > 0)
  val bankBytes: Int    = bankEntries * (bankBits / 8)
  val bytes:     BigInt = BigInt(banks) * bankBytes
}

/**
 * Per-tile topology: ordered cores. CPU kind is per-core; shared memory is tile-level.
 * CPU cores have system hart IDs; Ant contexts remain tile-local resources.
 */
case class TileTopology(
  param:         TileParam,
  main:          Boolean,
  cores:         Seq[TileCore],
  hartIds:       Seq[Int],
  executionIds:  Seq[Int],
  signatures:    Seq[BigInt],
  controller:    Option[Int],
  sharedStorage: Option[SharedStorageParams],
  coreFactories: Seq[String],
  factory:       String) {
  val cpuCoreIds: Seq[Int] = cores.indices.filter(i => !cores(i).isInstanceOf[AntTileCore])
  require(hartIds.size == cpuCoreIds.size && executionIds.size == cores.size)
  require(!main || controller.isEmpty, "The main tile has no task controller")
}
