package examples.poly.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.system.tile.WithBBTile
import framework.top.GlobalConfig
import examples.poly.configs.PolyConfig
import framework.builtin.configloader.ConfigLoader
import examples.poly.tiles.configs.TilesConfig

/**
 * Top-level Chipyard config fragment for the poly example.
 *
 * Same JSON-driven assembly pattern as toy / goban / konbi, but each
 * `tileConfig` entry can describe a different *kind* of tile — pure CPU,
 * a light DNN buckyball, an LLM buckyball, etc. Per-tile `withBuckyball`
 * is inferred from the `Option`s the tile assembler returns: a tile whose
 * cores are all `None` produces a Rocket-only BBTile, otherwise the
 * accelerator slots are wired up.
 *
 * NOTE: New architecture only supports single-Core single-SM per tile.
 * Multi-core (heterogeneous) configs are expanded to multiple single-core tiles.
 * TODO: Support multi-Core per tile when new architecture is extended.
 */
class WithNPolyTiles extends Config(WithNPolyTiles.assemble)

object WithNPolyTiles {

  def assemble: Parameters = {
    val nTiles      = PolyConfig().nTiles
    val tileConfigs = TilesConfig().tileConfigs
    require(
      tileConfigs.size == nTiles,
      s"poly tiles/configs/default.json lists ${tileConfigs.size} tile configs " +
        s"but poly/configs/default.json declares nTiles=$nTiles"
    )

    // New architecture: one WithBBTile per core (single-Core single-SM for now)
    val fragments: Seq[Config] = tileConfigs.flatMap { name =>
      val perCore = ConfigLoader.loadApply[Seq[Option[GlobalConfig]]](name)

      // Create one tile per core
      perCore.map { globalConfigOpt =>
        new WithBBTile(
          withBuckyball = globalConfigOpt.isDefined,
          globalConfig = globalConfigOpt.getOrElse(GlobalConfig())
        )
      }
    }

    fragments.reduce[Parameters](_ ++ _)
  }

}
