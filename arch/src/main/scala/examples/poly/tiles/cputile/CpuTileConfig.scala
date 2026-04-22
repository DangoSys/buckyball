package examples.poly.tiles.cputile

import framework.top.GlobalConfig
import framework.builtin.configloader.ConfigLoader
import examples.poly.tiles.cputile.configs.{CpuTileConfig => CpuTileParam}

/**
 * Pure CPU tile entry — no Buckyball accelerator on any core.
 *
 * Each entry in `coreConfigs` resolves reflectively to a `CpuCoreConfig`,
 * whose `apply()` returns `None` so the framework's BBTile drops the
 * accelerator slot and ties off RoCC.
 */
object CpuTileConfig {

  def apply(): Seq[Option[GlobalConfig]] = {
    val tileParam = CpuTileParam()
    tileParam.coreConfigs.map { name =>
      ConfigLoader.loadApply[Option[GlobalConfig]](name)
    }
  }

}
