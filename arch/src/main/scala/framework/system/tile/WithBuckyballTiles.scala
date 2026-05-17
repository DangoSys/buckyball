package framework.system.tile

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{CoherenceManagerWrapper, SubsystemBankedCoherenceKey}
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

    val tileFragments: Seq[Config] = topology.tiles.map { tile =>
      val resolved = if (withBuckyball) tile.cores else tile.cores.map(_ => None)
      new WithBBTile(
        withBuckyball = resolved.exists(_.isDefined),
        nCoresPerTile = tile.cores.size,
        buckyballPerCore = Some(resolved),
        privateDCache = tile.privateDCache
      )
    }

    // If any tile has privateDCache enabled, the per-tile InclusiveCache acts
    // as the last-level cache, which requires the system-level InclusiveCache
    // to be replaced with `incoherentManager`. InclusiveCache requires
    // `lastLevel = !managers.exists(_.regionType > UNCACHED)`, and a system-level
    // InclusiveCache downstream would convert DRAM's regionType to CACHED,
    // violating the constraint. See BBTile.scala for the full rationale.
    val anyPrivateDCache = topology.tiles.exists(_.privateDCache.isDefined)
    val coherenceFragment: Seq[Config] =
      if (anyPrivateDCache) Seq(new WithIncoherentSystemBus) else Nil

    (tileFragments ++ coherenceFragment).reduce[Parameters](_ ++ _)
  }

}

/**
 * Replaces the system-level coherence manager (default: InclusiveCache) with
 * `incoherentManager`, a name-only pass-through. Use when per-tile privateDCache
 * is enabled and acts as the last-level cache.
 *
 * Tile-to-tile coherence is NOT maintained at the system bus when this is
 * applied: software must use explicit cache management (flush/invalidate) or
 * uncacheable channels (scratchpad, DMA) for cross-tile communication.
 */
class WithIncoherentSystemBus
    extends Config((site, here, up) => {
      case SubsystemBankedCoherenceKey => up(SubsystemBankedCoherenceKey, site).copy(
          coherenceManager = CoherenceManagerWrapper.incoherentManager
        )
    })
