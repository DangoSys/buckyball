package examples.goban.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.system.tile.WithBBTile
import framework.top.GlobalConfig
import examples.goban.configs.GobanConfig
import framework.builtin.configloader.ConfigLoader
import examples.goban.tiles.configs.TilesConfig

/**
 * Top-level Chipyard config fragment for the goban example.
 *
 * Same JSON-driven assembly pattern as toy. Goban's distinguishing trait is
 * that each tile carries multiple homogeneous Buckyball cores; the per-tile
 * core list is described by the corresponding `Tile<i>Config` object.
 *
 * NOTE: New architecture only supports single-Core single-SM per tile.
 * Multi-core configs are expanded to multiple single-core tiles.
 * TODO: Support multi-Core per tile when new architecture is extended.
 */
class WithNGobanTiles(withBuckyball: Boolean = true) extends Config(WithNGobanTiles.assemble(withBuckyball))

object WithNGobanTiles {

  def assemble(withBuckyball: Boolean): Parameters = {
    val nTiles      = GobanConfig().nTiles
    val tileConfigs = TilesConfig().tileConfigs
    require(
      tileConfigs.size == nTiles,
      s"goban tiles/configs/default.json lists ${tileConfigs.size} tile configs " +
        s"but goban/configs/default.json declares nTiles=$nTiles"
    )

    // New architecture: one WithBBTile per core (single-Core single-SM for now)
    val fragments: Seq[Config] = tileConfigs.flatMap { name =>
      val perCore  = ConfigLoader.loadApply[Seq[Option[GlobalConfig]]](name)
      val resolved = if (withBuckyball) perCore else perCore.map(_ => None)

      // Create one tile per core
      resolved.map { globalConfigOpt =>
        new WithBBTile(
          withBuckyball = globalConfigOpt.isDefined,
          globalConfig = globalConfigOpt.getOrElse(GlobalConfig())
        )
      }
    }

    fragments.reduce[Parameters](_ ++ _)
  }

}
