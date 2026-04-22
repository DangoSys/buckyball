package examples.goban.tiles.gobantile.gobancore

import framework.top.GlobalConfig
import framework.balldomain.configs.BallDomainParam
import framework.builtin.configloader.ConfigLoader
import examples.goban.tiles.gobantile.gobancore.configs.{GobanCoreConfig => GobanCoreParam}

object GobanCoreConfig {

  def apply(): GlobalConfig = {
    val coreParam       = GobanCoreParam()
    val ballDomainParam = ConfigLoader.loadApply[BallDomainParam](coreParam.balldomain)
    GlobalConfig().copy(ballDomain = ballDomainParam)
  }

}
