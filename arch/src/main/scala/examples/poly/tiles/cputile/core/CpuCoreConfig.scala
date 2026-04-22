package examples.poly.tiles.cputile.core

import framework.top.GlobalConfig

/**
 * CPU-only core entry. Returns `None` so the framework's BBTile produces
 * a Rocket-only slot for this core (no Buckyball accelerator).
 *
 * Reading `core/configs/default.json` is intentionally skipped — the CPU
 * variant carries no Buckyball-side parameters, but the parameter
 * case class still exists for symmetry with other examples.
 */
object CpuCoreConfig {

  def apply(): Option[GlobalConfig] = None

}
