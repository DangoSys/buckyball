package framework.system.tile

import org.chipsalliance.cde.config.{Config, Parameters}
import framework.system.configloader.TomlConfigLoader

/**
 * Shared config fragment that builds an N-BBTile chipyard subsystem from a
 * TOML topology file.
 *
 * Replaces the per-example `WithNXxxTiles` classes — every example now just
 * picks a TOML path and calls this directly:
 *
 * {{{
 *   class BuckyballToyConfig extends Config(
 *     new WithBuckyballTiles("src/main/scala/examples/toy/toy.toml") ++
 *       new chipyard.config.WithSystemBusWidth(128) ++
 *       new sims.base.BuckyballBaseConfig
 *   )
 * }}}
 *
 * @param tomlPath      Path to the example's top-level TOML file (typically
 *                      `src/main/scala/examples/<name>/<name>.toml`).
 * @param withBuckyball When false, every Buckyball slot in the topology is
 *                      torn down (Rocket-only tiles). The tile/core counts
 *                      themselves are unchanged.
 */
class WithBuckyballTiles(
  tomlPath:      String,
  withBuckyball: Boolean = true)
    extends Config(WithBuckyballTiles.assemble(tomlPath, withBuckyball))

object WithBuckyballTiles {

  def assemble(tomlPath: String, withBuckyball: Boolean): Parameters = {
    val topology = TomlConfigLoader.load(tomlPath)

    val fragments: Seq[Config] = topology.tiles.map { tile =>
      val resolved = if (withBuckyball) tile.cores else tile.cores.map(_ => None)
      new WithBBTile(
        withBuckyball = resolved.exists(_.isDefined),
        nCoresPerTile = tile.cores.size,
        buckyballPerCore = Some(resolved),
        privateDCache = tile.privateDCache
      )
    }

    fragments.reduce[Parameters](_ ++ _)
  }

}
