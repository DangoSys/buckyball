package examples.toy.tiles.toytile.toycore

import framework.top.GlobalConfig
import framework.balldomain.configs.BallDomainParam
import framework.builtin.configloader.ConfigLoader
import examples.toy.tiles.toytile.toycore.configs.{ToyCoreConfig => ToyCoreParam}

/**
 * Toy core assembler.
 *
 * Reads `toycore/configs/default.json` to learn which Ball domain config
 * to use, dispatches to it reflectively, and overrides the framework
 * default `ballDomain` with the result.
 */
object ToyCoreConfig {

  def apply(): GlobalConfig = {
    val coreParam       = ToyCoreParam()
    val ballDomainParam = ConfigLoader.loadApply[BallDomainParam](coreParam.balldomain)
    GlobalConfig().copy(ballDomain = ballDomainParam)
  }

}
