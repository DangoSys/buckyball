package examples.toy.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.core.bbtile.WithBBTile
import framework.top.GlobalConfig
import examples.toy.configs.ToyConfig
import framework.builtin.configloader.ConfigLoader
import examples.toy.tiles.configs.TilesConfig

/**
 * Top-level Chipyard config fragment for the toy example.
 *
 * Fully JSON-driven: the tile/core/balldomain layout is described by the
 * `default.json` files under `examples/toy/...`. Each layer's JSON points
 * at a sibling object whose `apply()` produces the next layer's data;
 * this assembler reflectively walks that chain and emits one
 * `WithBBTile` fragment per tile.
 *
 * `withBuckyball = false` (used by `RocketOnlyConfig`) keeps the JSON
 * topology but tears down every Buckyball slot.
 */
class WithNToyTiles(withBuckyball: Boolean = true) extends Config(WithNToyTiles.assemble(withBuckyball))

object WithNToyTiles {

  def assemble(withBuckyball: Boolean): Parameters = {
    val nTiles      = ToyConfig().nTiles
    val tileConfigs = TilesConfig().tileConfigs
    require(
      tileConfigs.size == nTiles,
      s"tiles/configs/default.json lists ${tileConfigs.size} tile configs " +
        s"but toy/configs/default.json declares nTiles=$nTiles"
    )

    val fragments: Seq[Config] = tileConfigs.map { name =>
      val perCore  = ConfigLoader.loadApply[Seq[Option[GlobalConfig]]](name)
      val resolved = if (withBuckyball) perCore else perCore.map(_ => None)
      new WithBBTile(
        withBuckyball = resolved.exists(_.isDefined),
        nCoresPerTile = perCore.size,
        buckyballPerCore = Some(resolved)
      )
    }

    fragments.reduce[Parameters](_ ++ _)
  }

}
