package examples.poly.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.core.bbtile.WithBBTile
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

    val fragments: Seq[Config] = tileConfigs.map { name =>
      val perCore = ConfigLoader.loadApply[Seq[Option[GlobalConfig]]](name)
      new WithBBTile(
        withBuckyball = perCore.exists(_.isDefined),
        nCoresPerTile = perCore.size,
        buckyballPerCore = Some(perCore)
      )
    }

    fragments.reduce[Parameters](_ ++ _)
  }

}
