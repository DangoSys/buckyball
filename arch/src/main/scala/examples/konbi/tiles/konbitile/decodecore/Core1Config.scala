package examples.konbi.tiles.konbitile.decodecore

import framework.top.GlobalConfig
import framework.balldomain.configs.BallDomainParam
import framework.builtin.configloader.ConfigLoader
import examples.konbi.tiles.konbitile.decodecore.configs.KonbiCoreConfig

/** Decode core variant. */
object Core1Config {

  def apply(): GlobalConfig = {
    val coreParam       = KonbiCoreConfig()
    val ballDomainParam = ConfigLoader.loadApply[BallDomainParam](coreParam.balldomain)
    GlobalConfig().copy(ballDomain = ballDomainParam)
  }

}
