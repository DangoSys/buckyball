package examples.goban.tiles.gobantile

import framework.top.GlobalConfig
import framework.builtin.configloader.ConfigLoader
import examples.goban.tiles.gobantile.configs.{GobanTileConfig => GobanTileParam}

object GobanTileConfig {

  def apply(): Seq[Option[GlobalConfig]] = {
    val tileParam = GobanTileParam()
    val nCores    = tileParam.coreConfigs.size
    tileParam.coreConfigs.map { name =>
      val cfg = ConfigLoader.loadApply[GlobalConfig](name)
      Some(cfg.copy(top = cfg.top.copy(nCores = nCores)))
    }
  }

}
