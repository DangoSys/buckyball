package examples.konbi.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.system.tile.WithBBTile
import framework.top.GlobalConfig
import examples.konbi.configs.KonbiConfig
import framework.builtin.configloader.ConfigLoader
import examples.konbi.tiles.configs.TilesConfig

/**
 * Top-level Chipyard config fragment for the konbi example.
 *
 * Same JSON-driven assembly pattern as toy / goban. Konbi distinguishes
 * itself by carrying *heterogeneous* Buckyball cores within a tile (e.g.
 * prefill cores + decode cores); this is expressed by listing different
 * Core config objects in `tile-N/configs/default.json`.
 *
 * NOTE: New architecture only supports single-Core single-SM per tile.
 * Multi-core (heterogeneous) configs are expanded to multiple single-core tiles.
 * TODO: Support multi-Core per tile when new architecture is extended.
 */
class WithNKonbiTiles(withBuckyball: Boolean = true) extends Config(WithNKonbiTiles.assemble(withBuckyball))

object WithNKonbiTiles {

  def assemble(withBuckyball: Boolean): Parameters = {
    val nTiles      = KonbiConfig().nTiles
    val tileConfigs = TilesConfig().tileConfigs
    require(
      tileConfigs.size == nTiles,
      s"konbi tiles/configs/default.json lists ${tileConfigs.size} tile configs " +
        s"but konbi/configs/default.json declares nTiles=$nTiles"
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
