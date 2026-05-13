package examples.goban.tiles.gobantile

import framework.top.GlobalConfig
import framework.builtin.configloader.ConfigLoader
import examples.goban.tiles.gobantile.configs.{GobanTileConfig => GobanTileParam}

object GobanTile16CoreConfig {

  def apply(): Seq[Option[GlobalConfig]] = {
    val jsonStr   =
      scala.io.Source.fromFile("src/main/scala/examples/goban/tiles/gobantile/configs/tile16core.json").mkString
    val tileParam = upickle.default.read[GobanTileParam](jsonStr)
    val nCores    = tileParam.coreConfigs.size
    tileParam.coreConfigs.map { name =>
      val cfg = ConfigLoader.loadApply[GlobalConfig](name)
      Some(cfg.copy(top = cfg.top.copy(nCores = nCores)))
    }
  }

}
