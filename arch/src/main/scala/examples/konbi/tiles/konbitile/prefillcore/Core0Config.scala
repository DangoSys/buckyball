package examples.konbi.tiles.konbitile.prefillcore

import framework.top.GlobalConfig
import framework.balldomain.configs.BallDomainParam
import framework.builtin.configloader.ConfigLoader
import examples.konbi.tiles.konbitile.prefillcore.configs.KonbiCoreConfig

/** Prefill core variant. */
object Core0Config {

  def apply(): GlobalConfig = {
    val coreParam       = KonbiCoreConfig()
    val ballDomainParam = ConfigLoader.loadApply[BallDomainParam](coreParam.balldomain)
    GlobalConfig().copy(ballDomain = ballDomainParam)
  }

}
