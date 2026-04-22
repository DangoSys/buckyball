package examples.toy.tiles.toytile

import framework.top.GlobalConfig
import framework.builtin.configloader.ConfigLoader
import examples.toy.tiles.toytile.configs.{ToyTileConfig => ToyTileParam}

/**
 * Toy tile assembler.
 *
 * Reads `toytile/configs/default.json` to learn which Core configs to use,
 * dispatches to each reflectively, and produces the per-core
 * `Seq[Option[GlobalConfig]]` consumed by the framework's `WithBBTile`.
 *
 * Each per-core `GlobalConfig`'s `top.nCores` is patched here to equal the
 * number of cores in this tile, so that the framework's BBTile invariants
 * (`cfg.top.nCores == nCoresPerTile`) hold without each core having to
 * know how many siblings it has.
 */
object ToyTileConfig {

  def apply(): Seq[Option[GlobalConfig]] = {
    val tileParam = ToyTileParam()
    val nCores    = tileParam.coreConfigs.size
    tileParam.coreConfigs.map { name =>
      val cfg = ConfigLoader.loadApply[GlobalConfig](name)
      Some(cfg.copy(top = cfg.top.copy(nCores = nCores)))
    }
  }

}
