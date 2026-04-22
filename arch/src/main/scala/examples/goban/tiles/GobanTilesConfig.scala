package examples.goban.tiles

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.core.bbtile.WithBBTile
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
