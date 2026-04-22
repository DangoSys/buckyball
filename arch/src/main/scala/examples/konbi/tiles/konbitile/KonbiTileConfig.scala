package examples.konbi.tiles.konbitile

import framework.top.GlobalConfig
import framework.builtin.configloader.ConfigLoader
import examples.konbi.tiles.konbitile.configs.{KonbiTileConfig => KonbiTileParam}

object KonbiTileConfig {

  def apply(): Seq[Option[GlobalConfig]] = {
    val tileParam = KonbiTileParam()
    val nCores    = tileParam.coreConfigs.size
    tileParam.coreConfigs.map { name =>
      val cfg = ConfigLoader.loadApply[GlobalConfig](name)
      Some(cfg.copy(top = cfg.top.copy(nCores = nCores)))
    }
  }

}
